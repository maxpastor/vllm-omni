"""
Test streaming vs non-streaming codec codes consistency.

This script verifies that streaming and non-streaming generation produce
identical codec codes (token_id level comparison).

Usage:
    python test_streaming_vs_nonstreaming.py

    # Override model path
    MODEL_PATH=/your/path python test_streaming_vs_nonstreaming.py
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import Qwen3TTSModel

# Model path - can be overridden via MODEL_PATH environment variable
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")


def test_streaming_vs_nonstreaming_codes():
    """
    Test that streaming and non-streaming generation produce identical codec codes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    print("Model loaded successfully!")

    tts_model_type = model.model.tts_model_type
    print(f"Model type: {tts_model_type}")

    # Common parameters
    text = "这是一个流式生成测试的例子，我们来验证流式生成和批量生成的一致性。"
    chunk_size = 25
    max_new_tokens = 2048

    # Prepare common generation kwargs based on model type
    if tts_model_type == "custom_voice":
        speaker = "Vivian"
        language = "Chinese"
        instruct = "用清晰自然的语气说"
    elif tts_model_type == "voice_design":
        speaker = None
        language = "Chinese"
        instruct = "用清晰自然的语气说"
    elif tts_model_type == "base":
        speaker = None
        language = "Chinese"
        instruct = None
    else:
        raise ValueError(f"Unknown model type: {tts_model_type}")

    print("\n" + "=" * 60)
    print("CODEC CODES COMPARISON: Streaming vs Non-Streaming")
    print(f"  text={text!r}")
    print(f"  speaker={speaker!r}")
    print(f"  language={language!r}")
    print(f"  instruct={instruct!r}")
    print(f"  chunk_size={chunk_size}")
    print(f"  max_new_tokens={max_new_tokens}")
    print("  do_sample=False (greedy)")
    print("=" * 60)

    # ========== Prepare common inputs ==========
    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    print(f"\n[INPUT] input_ids shape: {input_ids[0].shape}")

    instruct_ids = [None]
    if instruct is not None and instruct != "":
        instruct_ids = [model._tokenize_texts([model._build_instruct_text(instruct)])[0]]
        print(f"[INPUT] instruct_ids shape: {instruct_ids[0].shape}")

    # Use greedy decoding for deterministic comparison
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.05,
    }
    print(f"[INPUT] gen_kwargs: {gen_kwargs}")

    # ========== NON-STREAMING: Get codec codes via generate() ==========
    print("\n" + "-" * 60)
    print(">>> NON-STREAMING: Generating codec codes with generate()...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    non_streaming_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[speaker],
        **gen_kwargs,
    )

    non_streaming_codes = non_streaming_codes_list[0]  # [T, K]
    print(f"[NON-STREAMING] codec_codes shape: {non_streaming_codes.shape}")
    print(f"[NON-STREAMING] Total tokens: {non_streaming_codes.shape[0]}")
    print(f"[NON-STREAMING] First 10 tokens (first codebook):\n  {non_streaming_codes[:10, 0].tolist()}")

    # ========== STREAMING: Get codec codes via generate_streaming_iter() ==========
    print("\n" + "-" * 60)
    print(">>> STREAMING: Generating codec codes with talker.generate_streaming_iter()...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Prepare talker inputs (same as generate_streaming does internally)
    talker_input_embeds, trailing_text_hiddens, tts_pad_embed = model.model._prepare_talker_inputs(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[speaker],
    )

    # Collect all codec codes from streaming iterator
    streaming_all_codes = []
    chunk_count = 0

    for chunk_output in model.model.talker.generate_streaming_iter(
        inputs_embeds=talker_input_embeds,
        attention_mask=torch.ones(
            (1, talker_input_embeds.shape[1]),
            device=talker_input_embeds.device,
            dtype=torch.long,
        ),
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        chunk_size=chunk_size,
        max_new_tokens=max_new_tokens,
        min_new_tokens=2,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=model.model.config.talker_config.codec_eos_token_id,
        suppress_tokens=[
            i
            for i in range(
                model.model.config.talker_config.vocab_size - 1024, model.model.config.talker_config.vocab_size
            )
            if i not in (model.model.config.talker_config.codec_eos_token_id,)
        ],
    ):
        chunk_count += 1
        streaming_all_codes.append(chunk_output.codec_codes)
        print(
            f"[STREAMING] Chunk {chunk_count}: codes shape={chunk_output.codec_codes.shape}, "
            f"total_generated={chunk_output.total_generated}, finished={chunk_output.is_finished}"
        )

    # Concatenate all streaming codes
    streaming_codes = torch.cat(streaming_all_codes, dim=0) if streaming_all_codes else torch.tensor([])
    print(f"\n[STREAMING] Total codec_codes shape: {streaming_codes.shape}")
    print(f"[STREAMING] Total tokens: {streaming_codes.shape[0]}")
    print(f"[STREAMING] First 10 tokens (first codebook):\n  {streaming_codes[:10, 0].tolist()}")

    # ========== COMPARE AND ASSERT (excluding EOS token) ==========
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS (TOKEN_ID LEVEL):")
    print("=" * 60)
    print(f"Non-streaming tokens (raw): {non_streaming_codes.shape[0]}")
    print(f"Streaming tokens (raw):     {streaming_codes.shape[0]}")

    # Move to CPU for comparison
    streaming_codes_cpu = streaming_codes.cpu()
    non_streaming_codes_cpu = non_streaming_codes.cpu()

    # Get EOS token ID
    eos_token_id = model.model.config.talker_config.codec_eos_token_id
    print(f"EOS token ID: {eos_token_id}")

    # Remove EOS tokens from the end (check first codebook for EOS)
    def remove_trailing_eos(codes, eos_id):
        """Remove trailing EOS tokens from codes."""
        if codes.shape[0] == 0:
            return codes
        # Check from the end, remove all trailing EOS tokens
        end_idx = codes.shape[0]
        while end_idx > 0 and codes[end_idx - 1, 0].item() == eos_id:
            end_idx -= 1
        return codes[:end_idx]

    non_streaming_codes_no_eos = remove_trailing_eos(non_streaming_codes_cpu, eos_token_id)
    streaming_codes_no_eos = remove_trailing_eos(streaming_codes_cpu, eos_token_id)

    print(f"Non-streaming tokens (no EOS): {non_streaming_codes_no_eos.shape[0]}")
    print(f"Streaming tokens (no EOS):     {streaming_codes_no_eos.shape[0]}")

    # Assert token counts match (excluding EOS)
    assert non_streaming_codes_no_eos.shape[0] == streaming_codes_no_eos.shape[0], (
        f"Token count mismatch (excluding EOS): non-streaming={non_streaming_codes_no_eos.shape[0]}, "
        f"streaming={streaming_codes_no_eos.shape[0]}"
    )
    print("✓ Token counts match (excluding EOS)!")

    # Assert all codebooks match
    num_codebooks = min(non_streaming_codes_no_eos.shape[1], streaming_codes_no_eos.shape[1])
    total_diff_count = 0
    first_diff_info = None

    for cb in range(num_codebooks):
        diff_mask = non_streaming_codes_no_eos[:, cb] != streaming_codes_no_eos[:, cb]
        diff_count = diff_mask.sum().item()

        if diff_count > 0:
            total_diff_count += diff_count
            if first_diff_info is None:
                diff_positions = torch.where(diff_mask)[0]
                pos = diff_positions[0].item()
                first_diff_info = {
                    "codebook": cb,
                    "position": pos,
                    "non_streaming": non_streaming_codes_no_eos[pos, cb].item(),
                    "streaming": streaming_codes_no_eos[pos, cb].item(),
                }

    if total_diff_count > 0:
        print(f"\n✗ Found {total_diff_count} token differences!")
        print(f"  First diff at codebook {first_diff_info['codebook']}, position {first_diff_info['position']}:")
        print(f"    Non-streaming: {first_diff_info['non_streaming']}")
        print(f"    Streaming:     {first_diff_info['streaming']}")

    assert total_diff_count == 0, (
        f"Token mismatch: {total_diff_count} differences found. "
        f"First diff at codebook {first_diff_info['codebook']}, position {first_diff_info['position']}: "
        f"non-streaming={first_diff_info['non_streaming']}, streaming={first_diff_info['streaming']}"
    )

    print(f"✓ All {non_streaming_codes_no_eos.shape[0]} tokens are IDENTICAL across all {num_codebooks} codebooks!")
    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_streaming_vs_nonstreaming_codes()
    sys.exit(0 if success else 1)
