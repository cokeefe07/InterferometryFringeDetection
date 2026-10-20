"""Reusable zoom & pan handler for a Tkinter canvas.

Provides right-click-and-drag panning and mouse-wheel zooming centered on
the pointer position. The handler is intentionally lightweight and
generic: it requires callbacks to get/set a numeric zoom level and to
trigger whatever redraw/rescale operation the host widget uses.

Usage example (from an object that has `viewport` canvas, `_zoom_level`,
and `_rescale_display_images` method):

    handler = ZoomPanHandler(
        widget=self.viewport,
        get_zoom=lambda: self._zoom_level,
        set_zoom=lambda z: setattr(self, '_zoom_level', z),
        rescale_callback=self._rescale_display_images,
    )

The handler attaches event bindings on construction and can be detached
with `handler.detach()` if needed.
"""
from typing import Callable, Optional
import logging
import os

# Configure logger
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'zoom_debug.log')

# Create a custom logger to avoid interfering with other loggers
logger = logging.getLogger('ZoomDebug')
logger.setLevel(logging.INFO)
# Check if handler already exists to avoid duplicate logs
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class ZoomPanHandler:
    def __init__(
        self,
        widget,
        get_zoom: Callable[[], float],
        set_zoom: Callable[[float], None],
        rescale_callback: Callable[[], None],
        min_zoom: float = 0.1,
        max_zoom: float = 10.0,
        zoom_step: float = 1.1,
    ):
        """Create and attach a zoom/pan handler to a Tk widget (usually Canvas).

        widget: a Tkinter widget (Canvas) that supports canvasx/canvasy,
                bbox('all'), xview_moveto, yview_moveto and update_idletasks.
        get_zoom/set_zoom: callbacks to read and write the numeric zoom level.
        rescale_callback: called after zoom level changes to redraw/rescale items.
        """
        self.widget = widget
        self.get_zoom = get_zoom
        self.set_zoom = set_zoom
        self.rescale_callback = rescale_callback
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.zoom_step = zoom_step

        # drag state
        self._is_dragging = False
        self._drag_start_root: Optional[tuple] = None
        self._drag_scroll_start_px: Optional[tuple] = None

        # zoom debounce state
        self._zoom_job = None
        self._pending_delta = 0
        self._pending_event = None

        # bind events (Windows MouseWheel event)
        try:
            self.widget.bind('<MouseWheel>', self._on_mousewheel)
            # Right button pan
            self.widget.bind('<Button-3>', self._on_pan_start)
            self.widget.bind('<B3-Motion>', self._on_pan_move)
            self.widget.bind('<ButtonRelease-3>', self._on_pan_end)
            
            logger.debug("ZoomPanHandler initialized and bindings attached.")
        except Exception:
            # Best-effort attach; failures will be silent so callers can handle
            pass

    def detach(self):
        """Remove event bindings created by this handler."""
        try:
            self.widget.unbind('<MouseWheel>')
            self.widget.unbind('<Button-3>')
            self.widget.unbind('<B3-Motion>')
            self.widget.unbind('<ButtonRelease-3>')
        except Exception:
            pass

    def _on_mousewheel(self, event):
        """Zoom in/out centered at mouse pointer position.

        Uses event.delta (Windows) sign to determine direction.
        Throttled to ensure intermediate frames render during rapid scrolling.
        """
        try:
            # Accumulate delta
            if hasattr(event, 'delta') and event.delta != 0:
                 self._pending_delta += event.delta
            
            # Store event for mouse coordinates
            self._pending_event = event
            
            # If a zoom is already scheduled, let it run (throttling)
            if self._zoom_job is not None:
                return

            # Schedule execution (30ms delay for responsiveness)
            self._zoom_job = self.widget.after(30, self._perform_zoom)
        except Exception as e:
            logger.error(f"Error in _on_mousewheel: {e}")

    def _perform_zoom(self):
        """Execute the actual zoom operation after debouncing."""
        self._zoom_job = None
        event = self._pending_event
        raw_delta = self._pending_delta
        self._pending_delta = 0
        
        if raw_delta == 0 or event is None:
            return

        try:
            # Normalize delta -- on Windows event.delta is multiple of 120
            delta = int(raw_delta / 120)
            if delta == 0:
                # Handle small deltas (e.g. trackpads) by forcing at least 1 step
                if raw_delta > 0: delta = 1
                elif raw_delta < 0: delta = -1
                else: return
            
            old_zoom = float(self.get_zoom())
            
            # Calculate zoom factor based on magnitude of delta
            steps = abs(delta)
            step_factor = self.zoom_step ** steps
            zoom_factor = step_factor if delta > 0 else (1.0 / step_factor)
            
            new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * zoom_factor))
            
            logger.debug(f"Zoom: {old_zoom:.4f} -> {new_zoom:.4f} (delta={delta}, raw={raw_delta})")

            if new_zoom == old_zoom:
                return

            # Mouse position relative to widget
            try:
                mouse_x = int(event.x)
                mouse_y = int(event.y)
            except Exception:
                # Fallback to center
                mouse_x = self.widget.winfo_width() // 2
                mouse_y = self.widget.winfo_height() // 2
            
            logger.debug(f"Mouse pos: ({mouse_x}, {mouse_y})")

            # Current bbox BEFORE redraw to derive anchor in image space
            bbox_before = self.widget.bbox('all')
            if bbox_before:
                width_before = float(bbox_before[2] - bbox_before[0])
                height_before = float(bbox_before[3] - bbox_before[1])
            else:
                width_before = 1.0
                height_before = 1.0
            left_px_before = float(self.widget.canvasx(0))
            top_px_before = float(self.widget.canvasy(0))
            
            logger.debug(f"Canvas offset before: ({left_px_before}, {top_px_before})")

            # Anchor in unscaled image coordinates (divide by old zoom)
            anchor_img_x = (left_px_before + float(mouse_x)) / max(old_zoom, 1e-9)
            anchor_img_y = (top_px_before + float(mouse_y)) / max(old_zoom, 1e-9)
            
            logger.debug(f"Anchor (img coords): ({anchor_img_x:.2f}, {anchor_img_y:.2f})")

            # Apply zoom & redraw
            self.set_zoom(new_zoom)
            try:
                self.rescale_callback()
            except Exception:
                pass
            try:
                self.widget.update_idletasks()
            except Exception:
                pass

            # Determine extents from scrollregion when available; fallback to bbox('all')
            try:
                sr = self.widget.cget('scrollregion')
            except Exception:
                sr = None
            if sr:
                try:
                    x0, y0, x1, y1 = map(float, str(sr).split())
                    min_x = x0; min_y = y0
                    width_after = max(1.0, x1 - x0)
                    height_after = max(1.0, y1 - y0)
                except Exception:
                    bbox_after = self.widget.bbox('all')
                    min_x = float(bbox_after[0]) if bbox_after else 0.0
                    min_y = float(bbox_after[1]) if bbox_after else 0.0
                    width_after = float(bbox_after[2] - bbox_after[0]) if bbox_after else 1.0
                    height_after = float(bbox_after[3] - bbox_after[1]) if bbox_after else 1.0
            else:
                bbox_after = self.widget.bbox('all')
                min_x = float(bbox_after[0]) if bbox_after else 0.0
                min_y = float(bbox_after[1]) if bbox_after else 0.0
                width_after = float(bbox_after[2] - bbox_after[0]) if bbox_after else 1.0
                height_after = float(bbox_after[3] - bbox_after[1]) if bbox_after else 1.0

            # Viewport size
            vp_w = max(1.0, float(self.widget.winfo_width()))
            vp_h = max(1.0, float(self.widget.winfo_height()))
            # Desired new top-left so that anchor stays under pointer
            anchor_abs_x_after = anchor_img_x * new_zoom
            anchor_abs_y_after = anchor_img_y * new_zoom
            new_left_px = anchor_abs_x_after - float(mouse_x)
            new_top_px = anchor_abs_y_after - float(mouse_y)
            
            logger.debug(f"Target new offset (px): ({new_left_px:.2f}, {new_top_px:.2f})")

            # Clamp to extents (allow negative minima)
            max_left = (min_x + width_after) - vp_w
            max_top = (min_y + height_after) - vp_h
            new_left_px = min(max(new_left_px, min_x), max_left if max_left > min_x else min_x)
            new_top_px = min(max(new_top_px, min_y), max_top if max_top > min_y else min_y)
            
            logger.debug(f"Clamped new offset (px): ({new_left_px:.2f}, {new_top_px:.2f})")
            logger.debug(f"Scroll region/BBox: min_x={min_x}, min_y={min_y}, w={width_after}, h={height_after}")

            # Convert to fractions relative to scrollregion
            new_x_fraction = (new_left_px - min_x) / width_after if width_after > 0 else 0.0
            new_y_fraction = (new_top_px - min_y) / height_after if height_after > 0 else 0.0
            
            logger.debug(f"Moveto fractions: ({new_x_fraction:.4f}, {new_y_fraction:.4f})")

            try:
                self.widget.xview_moveto(max(0.0, min(1.0, new_x_fraction)))
                self.widget.yview_moveto(max(0.0, min(1.0, new_y_fraction)))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error in _perform_zoom: {e}")
            # swallow exceptions to avoid breaking UI
            pass

    def _get_scroll_offsets_px(self):
        try:
            sr = None
            try:
                sr = self.widget.cget('scrollregion')
            except Exception:
                sr = None
            if sr:
                x0, y0, x1, y1 = map(float, str(sr).split())
                min_x = x0; min_y = y0
                width = max(1.0, x1 - x0)
                height = max(1.0, y1 - y0)
            else:
                bbox = self.widget.bbox('all')
                if not bbox:
                    return 0.0, 0.0, 1.0, 1.0
                min_x = float(bbox[0]); min_y = float(bbox[1])
                width = float(bbox[2] - bbox[0])
                height = float(bbox[3] - bbox[1])
            x_frac0 = self.widget.xview()[0]
            y_frac0 = self.widget.yview()[0]
            return min_x + x_frac0 * width, min_y + y_frac0 * height, width, height
        except Exception:
            return 0.0, 0.0, 1.0, 1.0

    def _on_pan_start(self, event):
        try:
            self._is_dragging = True
            # record root coords so movement works even if cursor leaves widget
            self._drag_start_root = (event.x_root, event.y_root)
            sx, sy, _, _ = self._get_scroll_offsets_px()
            self._drag_scroll_start_px = (sx, sy)
            try:
                # set a panning cursor if widget supports configure
                self.widget.configure(cursor='fleur')
            except Exception:
                pass
            return 'break'
        except Exception:
            return None

    def _on_pan_move(self, event):
        if not self._is_dragging:
            return None
        try:
            dx = event.x_root - self._drag_start_root[0]
            dy = event.y_root - self._drag_start_root[1]
            start_x_px, start_y_px = self._drag_scroll_start_px
            new_x_px = start_x_px - dx
            new_y_px = start_y_px - dy
            # Determine extents and normalize to fractions relative to min
            try:
                sr = self.widget.cget('scrollregion')
            except Exception:
                sr = None
            if sr:
                try:
                    x0, y0, x1, y1 = map(float, str(sr).split())
                    min_x = x0; min_y = y0
                    width = max(1.0, x1 - x0)
                    height = max(1.0, y1 - y0)
                except Exception:
                    bbox = self.widget.bbox('all')
                    min_x = float(bbox[0]) if bbox else 0.0
                    min_y = float(bbox[1]) if bbox else 0.0
                    width = float(bbox[2] - bbox[0]) if bbox else 1.0
                    height = float(bbox[3] - bbox[1]) if bbox else 1.0
            else:
                bbox = self.widget.bbox('all')
                min_x = float(bbox[0]) if bbox else 0.0
                min_y = float(bbox[1]) if bbox else 0.0
                width = float(bbox[2] - bbox[0]) if bbox else 1.0
                height = float(bbox[3] - bbox[1]) if bbox else 1.0
            new_x_frac = 0.0 if width <= 0 else max(0.0, min(1.0, (new_x_px - min_x) / width))
            new_y_frac = 0.0 if height <= 0 else max(0.0, min(1.0, (new_y_px - min_y) / height))
            try:
                self.widget.xview_moveto(new_x_frac)
                self.widget.yview_moveto(new_y_frac)
            except Exception:
                pass
            return 'break'
        except Exception:
            return None

    def _on_pan_end(self, event):
        try:
            self._is_dragging = False
            try:
                self.widget.configure(cursor='')
            except Exception:
                pass
            return 'break'
        except Exception:
            return None


__all__ = ['ZoomPanHandler']
