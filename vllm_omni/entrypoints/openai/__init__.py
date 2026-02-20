# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OpenAI-compatible API entrypoints for vLLM-Omni.

Provides:
- omni_run_server: Main server entry point (auto-detects model type)
- OmniOpenAIServingChat: Unified chat completion handler for both LLM and diffusion models
"""

try:
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
except ImportError:
    # Some chat-serving imports are not compatible with vllm<=0.15.x.
    # Speech-serving modules should still remain importable.
    OmniOpenAIServingChat = None

try:
    from vllm_omni.entrypoints.openai.api_server import (
        build_async_omni,
        omni_init_app_state,
        omni_run_server,
    )
except ImportError:
    # vllm<=0.15.x does not expose some api_server dependencies.
    # Keep speech-serving imports usable even when API-server wiring is unavailable.
    build_async_omni = None
    omni_init_app_state = None
    omni_run_server = None

__all__ = [
    # Serving classes
]

if OmniOpenAIServingChat is not None:
    __all__.append("OmniOpenAIServingChat")

if omni_run_server is not None:
    __all__.extend(
        [
            "omni_run_server",
            "build_async_omni",
            "omni_init_app_state",
        ]
    )
