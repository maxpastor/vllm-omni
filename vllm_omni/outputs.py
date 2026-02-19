import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from vllm.outputs import RequestOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.inputs.data import OmniPromptType


class OmniModelRunnerOutput(ModelRunnerOutput):
    """Model runner output for omni models.

    Extends the base ModelRunnerOutput with support for multimodal outputs
    that may be produced by non-autoregressive stages.

    Attributes:
        multimodal_outputs: Optional dictionary mapping modality names to
            output tensors (e.g., {"image": tensor, "audio": tensor})
    """

    multimodal_outputs: dict[str, torch.Tensor] | None = None
    # IDs of requests whose KV cache has been extracted from GPU/NPU to CPU.
    # The Scheduler can safely free the block tables for these requests.
    kv_extracted_req_ids: list[str] | None = None


@dataclass
class OmniRequestOutput:
    """Unified request output for both pipeline stages and diffusion models.

    This class handles outputs from:
    1. Multi-stage LLM pipelines (with stage_id, final_output_type, request_output)
    2. Diffusion models (with images, prompt, metrics)

    Attributes:
        request_id: Unique identifier for this request
        finished: Whether generation is complete
        stage_id: Identifier of the stage that produced this output (pipeline mode)
        final_output_type: Type of output ("text", "image", "audio", "latents")
        request_output: The underlying RequestOutput from the stage (pipeline mode)
        images: List of generated PIL images (diffusion mode)
        prompt: The prompt used for generation (diffusion mode)
        latents: Optional tensor of latent representations (diffusion mode)
        metrics: Optional dictionary of generation metrics
    """

    request_id: str = ""
    finished: bool = True

    # Pipeline stage fields
    stage_id: int | None = None
    final_output_type: str = "text"
    request_output: RequestOutput | None = None

    # Diffusion model fields
    images: list[Image.Image] = field(default_factory=list)
    prompt: OmniPromptType | None = None
    latents: torch.Tensor | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    _multimodal_output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pipeline(
        cls,
        stage_id: int,
        final_output_type: str,
        request_output: RequestOutput,
    ) -> "OmniRequestOutput":
        """Create output from pipeline stage.

        Args:
            stage_id: Stage identifier
            final_output_type: Type of output
            request_output: The stage's output

        Returns:
            OmniRequestOutput configured for pipeline mode
        """
        return cls(
            request_id=getattr(request_output, "request_id", ""),
            stage_id=stage_id,
            final_output_type=final_output_type,
            request_output=request_output,
            finished=True,
        )

    @classmethod
    def from_diffusion(
        cls,
        request_id: str,
        images: list[Image.Image],
        prompt: OmniPromptType | None = None,
        metrics: dict[str, Any] | None = None,
        latents: torch.Tensor | None = None,
        multimodal_output: dict[str, Any] | None = None,
        final_output_type: str = "image",
    ) -> "OmniRequestOutput":
        """Create output from diffusion model.

        Args:
            request_id: Request identifier
            images: Generated images
            prompt: The prompt used
            metrics: Generation metrics
            latents: Optional latent tensors

        Returns:
            OmniRequestOutput configured for diffusion mode
        """
        return cls(
            request_id=request_id,
            final_output_type=final_output_type,
            images=images,
            prompt=prompt,
            latents=latents,
            metrics=metrics or {},
            _multimodal_output=multimodal_output or {},
            finished=True,
        )

    @property
    def multimodal_output(self) -> dict[str, Any]:
        """Return multimodal output from the underlying request output or local field.

        For pipeline outputs, this checks completion outputs first, then request_output.
        For diffusion outputs, this returns the local _multimodal_output field.
        """
        if self.request_output is not None:
            # Check completion outputs first (where multimodal_output is attached)
            if self.request_output.outputs:
                for output in self.request_output.outputs:
                    mm = getattr(output, "multimodal_output", None)
                    if mm:
                        return mm
            return getattr(self.request_output, "multimodal_output", {})
        return self._multimodal_output

    @property
    def num_images(self) -> int:
        """Return the number of generated images."""
        return len(self.images)

    # Pass-through properties keep vLLM serving codepaths compatible with
    # OmniRequestOutput for pipeline outputs (Issue #345).
    @property
    def prompt_token_ids(self) -> list[int] | None:
        """Return prompt token IDs from the underlying request output.

        This property is required for compatibility with vLLM's streaming
        chat completion generator which checks res.prompt_token_ids.
        """
        if self.request_output is not None:
            return getattr(self.request_output, "prompt_token_ids", None)
        return None

    @property
    def outputs(self) -> list[Any]:
        """Return outputs from the underlying request output.

        This property is required for compatibility with vLLM's streaming
        and non-streaming chat completion generators.
        """
        if self.request_output is not None:
            return getattr(self.request_output, "outputs", [])
        return []

    @property
    def encoder_prompt_token_ids(self) -> list[int] | None:
        """Return encoder prompt token IDs from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "encoder_prompt_token_ids", None)
        return None

    @property
    def prompt_logprobs(self) -> Any:
        """Return prompt logprobs from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "prompt_logprobs", None)
        return None

    @property
    def num_cached_tokens(self) -> int | None:
        """Return number of cached tokens from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "num_cached_tokens", None)
        return None

    @property
    def kv_transfer_params(self) -> Any:
        """Return KV transfer params from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "kv_transfer_params", None)
        return None

    @property
    def is_diffusion_output(self) -> bool:
        """Check if this is a diffusion model output."""
        return len(self.images) > 0 or self.final_output_type == "image"

    @property
    def is_pipeline_output(self) -> bool:
        """Check if this is a pipeline stage output."""
        return self.stage_id is not None and self.request_output is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "finished": self.finished,
            "final_output_type": self.final_output_type,
        }

        if self.is_diffusion_output:
            result.update(
                {
                    "num_images": self.num_images,
                    "prompt": self.prompt,
                    "metrics": self.metrics,
                }
            )

        if self.is_pipeline_output:
            result.update(
                {
                    "stage_id": self.stage_id,
                }
            )

        return result

    def __repr__(self) -> str:
        """Custom repr to properly show image count instead of image objects."""
        # For images, show count instead of full list
        images_repr = f"[{len(self.images)} PIL Images]" if self.images else "[]"

        # Build repr string
        parts = [
            f"request_id={self.request_id!r}",
            f"finished={self.finished}",
            f"stage_id={self.stage_id}",
            f"final_output_type={self.final_output_type!r}",
            f"request_output={self.request_output}",
            f"images={images_repr}",
            f"prompt={self.prompt!r}",
            f"latents={self.latents}",
            f"metrics={self.metrics}",
            f"multimodal_output={self._multimodal_output}",
        ]

        return f"OmniRequestOutput({', '.join(parts)})"


@dataclass
class StreamingChunkOutput:
    """Output for each streaming chunk during TTS generation."""

    codec_codes: torch.Tensor  # [chunk_size, num_quantizers] codec tokens for this chunk
    hidden_states: torch.Tensor | None = None  # corresponding hidden states
    chunk_idx: int = 0  # chunk index
    is_finished: bool = False  # whether generation is complete
    total_generated: int = 0  # total tokens generated so far


class AsyncDecodingPipeline:
    """
    Asynchronous decoding pipeline that runs audio decoding in a background thread
    while generation continues in the main thread.
    """

    def __init__(
        self,
        speech_tokenizer,
        ref_code: torch.Tensor | None = None,
        left_context_size: int = 25,
        max_queue_size: int = 10,
    ):
        self.speech_tokenizer = speech_tokenizer
        self.ref_code = ref_code
        self.left_context_size = left_context_size

        # Queue for codec chunks to be decoded
        # Each item is (codes_with_context, is_last, context_frames_to_remove)
        self._input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        # Queue for decoded audio chunks
        self._output_queue: queue.Queue = queue.Queue()

        self._decode_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._all_codes: list[torch.Tensor] = []
        self._sample_rate: int | None = None

    def start(self):
        """Start the background decoding thread."""
        if self._started:
            return
        self._stop_event.clear()
        self._decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._decode_thread.start()
        self._started = True

    def _decode_worker(self):
        """Background worker that decodes codec chunks to audio."""
        chunk_idx = 0

        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:  # Sentinel to stop
                break

            codes_chunk, is_last, context_frames = item

            # Decode the chunk
            try:
                # codes shape: [seq_len, num_quantizers] -> [1, seq_len, num_quantizers]
                # model.decode expects [B, T, K] and internally transposes to [B, K, T]
                codes_for_decode = codes_chunk.unsqueeze(0)
                wavs, sr = self.speech_tokenizer.decode({"audio_codes": codes_for_decode})
                audio_chunk = wavs[0]  # numpy array
                self._sample_rate = sr

                # Remove context samples from the beginning of the audio
                if context_frames > 0:
                    upsample_rate = getattr(self.speech_tokenizer.model, "decode_upsample_rate", 2000)
                    context_samples = context_frames * upsample_rate
                    if context_samples < len(audio_chunk):
                        audio_chunk = audio_chunk[context_samples:]

                self._output_queue.put((audio_chunk, is_last, sr, None))
            except Exception as e:
                self._output_queue.put((None, is_last, None, e))

            chunk_idx += 1

    def submit_chunk(self, codec_codes: torch.Tensor, is_last: bool = False):
        """Submit a chunk of codec codes for decoding."""
        self._all_codes.append(codec_codes)

        # Prepare chunk with context and track how many context frames were added
        context_frames = 0

        if len(self._all_codes) == 1:
            # First chunk - prepend ref_code if available
            if self.ref_code is not None:
                codes_with_context = torch.cat([self.ref_code, codec_codes], dim=0)
                context_frames = self.ref_code.shape[0]
            else:
                codes_with_context = codec_codes
                context_frames = 0
        else:
            # Subsequent chunks - add left context from previously generated codes
            context_codes = torch.cat(self._all_codes[:-1], dim=0)
            context_frames = min(self.left_context_size, context_codes.shape[0])
            context_start = context_codes.shape[0] - context_frames
            context = context_codes[context_start:]
            codes_with_context = torch.cat([context, codec_codes], dim=0)

        self._input_queue.put((codes_with_context, is_last, context_frames))

        # Limit memory usage: only keep enough codes for left_context_size
        # Merge old codes if we have too many chunks
        if len(self._all_codes) > 10:
            # Merge all codes and keep only the last left_context_size frames
            all_merged = torch.cat(self._all_codes, dim=0)
            if all_merged.shape[0] > self.left_context_size:
                self._all_codes = [all_merged[-self.left_context_size :]]
            else:
                self._all_codes = [all_merged]

    def get_decoded_chunk(self, timeout: float | None = None) -> tuple[Any, bool, int | None, Exception | None]:
        """
        Get the next decoded audio chunk.

        Returns:
            tuple: (audio_chunk, is_last, sample_rate, error)
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None, False, None, None

    def iter_decoded_chunks(self) -> Iterator[tuple[Any, bool, int]]:
        """Iterate over decoded audio chunks as they become available."""
        while True:
            audio, is_last, sr, error = self.get_decoded_chunk(timeout=1.0)
            if error is not None:
                raise error
            if audio is not None:
                yield audio, is_last, sr
            if is_last:
                break

    def stop(self):
        """Stop the decoding pipeline."""
        self._stop_event.set()
        self._input_queue.put(None)  # Sentinel
        if self._decode_thread is not None:
            self._decode_thread.join(timeout=2.0)
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
