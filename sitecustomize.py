"""Automatic runtime configuration for LiteRT-LM EmbeddingEngine."""

import os
import litert_lm

_orig_init = litert_lm.EmbeddingEngine.__init__


def _patched_init(self, *args, **kwargs):
  # Set default max_input_length to 8192 tokens (full context window)
  kwargs.setdefault(
      "max_input_length",
      int(os.environ.get("LITERT_LM_MAX_INPUT_LENGTH", "8192")),
  )
  # Set default vision tokens per frame to 140 (maximum vision signature)
  kwargs.setdefault(
      "vision_tokens_per_image",
      int(os.environ.get("LITERT_LM_VISION_TOKENS", "140")),
  )
  return _orig_init(self, *args, **kwargs)


# Hook into EmbeddingEngine initialization
litert_lm.EmbeddingEngine.__init__ = _patched_init
