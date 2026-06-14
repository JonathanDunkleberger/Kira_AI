# kira.storytime — pre-generated shadow-puppet "Storytime" shows.
#
# GENERATE-THEN-PERFORM, review-gated: Claude writes a script → segmented beats →
# Gemini (Nano-Banana) pre-generates one silhouette image per beat in a locked
# style → the operator reviews/re-rolls in the dashboard → the show performs
# (scene swaps on the overlay synced to Kira's TTS narration).

from .show import StorytimeShow
from .image_client import ImageProvider, get_image_provider, ImageGenError

__all__ = ["StorytimeShow", "ImageProvider", "get_image_provider", "ImageGenError"]
