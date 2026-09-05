"""
Imprint - ComfyUI Custom Nodes
Blockchain-backed provenance logging for AI-generated content

Copy this entire folder to: ComfyUI/custom_nodes/imprint/
Then restart ComfyUI.
"""

from .imprint_nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    __version__,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
