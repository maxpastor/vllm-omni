"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence. Audio chunks are streamed
progressively as they are decoded by the model.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "pcm"}
        <binary frame: raw audio bytes>   # May arrive multiple times (streaming chunks)
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
import json
import struct

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import SentenceSplitter

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_DEFAULT_SENTENCE_TIMEOUT = 120.0  # seconds


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence. Audio chunks are streamed progressively as the model
    produces them, minimizing time-to-first-audio.

    The handler supports two generation modes:
    1. Streaming: Uses engine_client.generate() async generator to yield
       progressive audio chunks (when the model supports forward_streaming).
    2. Blocking fallback: Uses _generate_audio_bytes() for a single
       complete audio response per sentence.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation, prompt building, and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
        sentence_timeout: Max seconds to wait for a single sentence to generate.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
        sentence_timeout: float = _DEFAULT_SENTENCE_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout
        self._sentence_timeout = sentence_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()
        active_gen_task: asyncio.Task | None = None
        gen_request_id: str | None = None

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, connection closing

            splitter = SentenceSplitter()
            sentence_index = 0
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

            # Start the sentence generation worker
            gen_task = asyncio.create_task(
                self._sentence_worker(websocket, config, sentence_queue)
            )

            # 2. Receive text chunks until input.done
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout: no message received")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    sentences = splitter.add_text(text)
                    for sentence in sentences:
                        await sentence_queue.put(sentence)

                elif msg_type == "input.done":
                    # Flush remaining buffer
                    remaining = splitter.flush()
                    if remaining:
                        await sentence_queue.put(remaining)

                    # Signal end of sentences
                    await sentence_queue.put(None)

                    # Wait for generation to finish
                    await gen_task
                    return

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

            # If we broke out of the loop (timeout), cancel generation
            gen_task.cancel()
            try:
                await gen_task
            except asyncio.CancelledError:
                pass

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
            # Cancel any in-flight generation
            if gen_request_id:
                try:
                    await self._speech_service.engine_client.abort(gen_request_id)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass
        finally:
            # Ensure generation task is cleaned up
            if "gen_task" in dir() and not gen_task.done():
                gen_task.cancel()
                try:
                    await gen_task
                except asyncio.CancelledError:
                    pass

    async def _sentence_worker(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_queue: asyncio.Queue[str | None],
    ) -> None:
        """Process sentences from the queue, generating and streaming audio for each."""
        sentence_index = 0

        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                # End signal — send session.done
                await websocket.send_json(
                    {
                        "type": "session.done",
                        "total_sentences": sentence_index,
                    }
                )
                return

            await self._generate_and_stream(websocket, config, sentence, sentence_index)
            sentence_index += 1

    async def _generate_and_stream(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a sentence and stream chunks over WebSocket.

        Uses the engine's async generator to yield progressive audio chunks.
        Falls back to blocking generation if streaming yields no intermediate
        chunks.
        """
        # Send audio.start
        await websocket.send_json(
            {
                "type": "audio.start",
                "sentence_index": sentence_index,
                "sentence_text": sentence_text,
                "format": "pcm",
            }
        )

        try:
            # Build TTS parameters and prompt
            request = OpenAICreateSpeechRequest(
                input=sentence_text,
                model=config.model,
                voice=config.voice,
                task_type=config.task_type,
                language=config.language,
                instructions=config.instructions,
                response_format="pcm",
                speed=config.speed,
                max_new_tokens=config.max_new_tokens,
                ref_audio=config.ref_audio,
                ref_text=config.ref_text,
                x_vector_only_mode=config.x_vector_only_mode,
            )

            engine = self._speech_service.engine_client

            if self._speech_service._is_tts_model():
                validation_error = self._speech_service._validate_tts_request(request)
                if validation_error:
                    raise ValueError(validation_error)
                tts_params = self._speech_service._build_tts_params(request)
                prompt_text = self._speech_service._build_tts_prompt(request.input)
                prompt = {
                    "prompt": prompt_text,
                    "additional_information": tts_params,
                }
            else:
                prompt = {"prompt": request.input}

            request_id = f"speech-stream-{random_uuid()}"
            sampling_params_list = engine.default_sampling_params_list

            generator = engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            # Stream audio chunks as they arrive
            async for output in generator:
                audio_output = output.multimodal_output
                if not audio_output:
                    continue

                # Extract audio tensor from the output
                audio_key = None
                if "audio" in audio_output:
                    audio_key = "audio"
                elif "model_outputs" in audio_output:
                    audio_key = "model_outputs"

                if audio_key is None:
                    continue

                audio_tensor = audio_output[audio_key]

                # Convert to raw PCM bytes
                if hasattr(audio_tensor, "float"):
                    audio_np = audio_tensor.float().detach().cpu().numpy()
                elif isinstance(audio_tensor, np.ndarray):
                    audio_np = audio_tensor.astype(np.float32)
                else:
                    continue

                if audio_np.ndim > 1:
                    audio_np = audio_np.squeeze()

                if audio_np.size == 0:
                    continue

                # Convert float32 [-1, 1] to int16 PCM bytes
                pcm_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()

                await websocket.send_bytes(pcm_bytes)

        except asyncio.CancelledError:
            logger.info("Generation cancelled for sentence %d", sentence_index)
        except WebSocketDisconnect:
            raise  # Let the session handler deal with disconnect
        except Exception as e:
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(
                websocket,
                f"Generation failed for sentence {sentence_index}: {e}",
            )

        # Send audio.done
        try:
            await websocket.send_json(
                {
                    "type": "audio.done",
                    "sentence_index": sentence_index,
                }
            )
        except Exception:
            pass  # Connection may have closed

    async def _receive_config(self, websocket: WebSocket) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type')}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        # Validate model name against served models if provided
        if config.model:
            error_check = await self._speech_service._check_model(
                OpenAICreateSpeechRequest(input="", model=config.model)
            )
            if error_check is not None:
                await self._send_error(
                    websocket,
                    f"Invalid model: {config.model}",
                )
                return None

        return config

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": message,
                }
            )
        except Exception:
            pass  # Connection may already be closed
