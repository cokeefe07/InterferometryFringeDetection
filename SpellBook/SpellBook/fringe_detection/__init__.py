"""fringe_detection package: moved helpers from repository root."""

from .shading_pipeline import pipeline_shading_sauvola, read_gray
from .fringe_utils import binarize, line_kernel, oriented_opening, overlay_mask_on_gray, remove_humps, remove_branches, fill_holes, remove_steep_segments
from .ui_helpers import make_slider_row, to_photoimage_from_bgr_with_scale

__all__ = [
    'pipeline_shading_sauvola', 'read_gray',
    'binarize', 'line_kernel', 'oriented_opening', 'overlay_mask_on_gray', 'remove_humps', 'remove_branches', 'fill_holes', 'remove_steep_segments',
    'make_slider_row', 'to_photoimage_from_bgr_with_scale'
]
