"""Universal zoom & pan handlers (editor-style) for Tkinter canvases.

Provides two lightweight classes:

OffsetZoomPan:
    For canvases where the host manages an (offset, scale) and redraws the
    image manually (like the fringe editor). It preserves cursor position
    while zooming and updates the offset directly.

ScrollZoomPan:
    For canvases using a scrollregion with xview/yview (like the detection
    viewport). It adjusts the zoom variable and scroll fractions to keep the
    cursor position stable.

Both classes:
    - Right mouse button drag pans.
    - Mouse wheel zooms toward cursor (ignore when Ctrl pressed so the
      caller can use Ctrl+Wheel for other shortcuts).
    - Configurable min/max zoom and wheel factor.

Usage (Editor-style):
    handler = OffsetZoomPan(
        widget=self.canvas,
        get_zoom=lambda: self._zoom,
        set_zoom=lambda z: setattr(self, '_zoom', z),
        get_offset=lambda: (self._offset if self._offset is not None else getattr(self, '_img_topleft', (0,0))),
        set_offset=lambda pt: setattr(self, '_offset', pt),
        redraw=lambda: self._refresh_display(False),
    )

Usage (Scrollregion-style):
    handler = ScrollZoomPan(
        widget=self.viewport,
        get_zoom=lambda: self._zoom_level,
        set_zoom=lambda z: setattr(self, '_zoom_level', z),
        redraw=self._rescale_display_images,
    )

You can call .detach() to remove bindings if needed.
"""
from __future__ import annotations

from typing import Callable, Tuple
import tkinter as tk


class _Base:
    def __init__(self, widget: tk.Widget):
        self.widget = widget
        self._dragging = False
        self._drag_root_start: Tuple[int,int] | None = None
        self._scroll_start_px: Tuple[float,float] | None = None

    # Common guards
    @staticmethod
    def _is_ctrl(event) -> bool:
        try:
            return bool(getattr(event, 'state', 0) & 0x4)  # Control mask
        except Exception:
            return False

    def detach(self):  # Overridden per subclass
        pass


class OffsetZoomPan(_Base):
    """Zoom/pan handler for offset-based redraw model."""
    def __init__(
        self,
        widget: tk.Widget,
        get_zoom: Callable[[], float],
        set_zoom: Callable[[float], None],
        get_offset: Callable[[], Tuple[int,int]],
        set_offset: Callable[[Tuple[int,int]], None],
        redraw: Callable[[], None],
        min_zoom: float = 0.1,
        max_zoom: float = 16.0,
        wheel_factor: float = 1.1,
    ):
        super().__init__(widget)
        self.get_zoom = get_zoom
        self.set_zoom = set_zoom
        self.get_offset = get_offset
        self.set_offset = set_offset
        self.redraw = redraw
        self.min_zoom = float(min_zoom)
        self.max_zoom = float(max_zoom)
        self.wheel_factor = float(wheel_factor)
        self._bind()

    def _bind(self):
        try:
            self.widget.bind('<MouseWheel>', self._on_wheel, add='+')
            self.widget.bind('<Button-3>', self._on_pan_start, add='+')
            self.widget.bind('<B3-Motion>', self._on_pan_move, add='+')
            self.widget.bind('<ButtonRelease-3>', self._on_pan_end, add='+')
        except Exception:
            pass

    def detach(self):
        for seq in ('<MouseWheel>', '<Button-3>', '<B3-Motion>', '<ButtonRelease-3>'):
            try:
                self.widget.unbind(seq)
            except Exception:
                pass

    def _on_wheel(self, event):
        if self._is_ctrl(event):  # allow caller to use Ctrl+Wheel for other actions
            return
        try:
            delta = int(getattr(event, 'delta', 0))
        except Exception:
            delta = 0
        if delta == 0:
            return
        old_zoom = float(self.get_zoom())
        factor = self.wheel_factor if delta > 0 else (1.0 / self.wheel_factor)
        new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        # Cursor in widget coords (derive from root so handler can be bound elsewhere)
        try:
            x_c = int(getattr(event, 'x_root')) - int(self.widget.winfo_rootx())
            y_c = int(getattr(event, 'y_root')) - int(self.widget.winfo_rooty())
        except Exception:
            x_c = int(getattr(event, 'x', self.widget.winfo_width() // 2))
            y_c = int(getattr(event, 'y', self.widget.winfo_height() // 2))
        # Retrieve current offset
        ox, oy = self.get_offset()
        # Effective scale before/after = base_scale * zoom. Host may recompute
        # base scale internally; we infer relative zoom so anchor position stays.
        # Use simple proportional method: treat offset + (x_c - ox)/old_zoom as image coord.
        # This relies on host redrawing with new zoom then we set new offset that keeps the point.
        if old_zoom <= 0:
            old_zoom = 1e-6
        img_x = (x_c - ox) / old_zoom
        img_y = (y_c - oy) / old_zoom
        self.set_zoom(new_zoom)
        try:
            self.redraw()
        except Exception:
            pass
        new_ox = int(round(x_c - img_x * new_zoom))
        new_oy = int(round(y_c - img_y * new_zoom))
        self.set_offset((new_ox, new_oy))
        try:
            self.redraw()
        except Exception:
            pass

    def _on_pan_start(self, event):
        self._dragging = True
        self._drag_root_start = (int(event.x_root), int(event.y_root))
        self._start_offset = self.get_offset()
        try:
            self.widget.configure(cursor='fleur')
        except Exception:
            pass
        return 'break'

    def _on_pan_move(self, event):
        if not self._dragging:
            return
        dx = int(event.x_root) - self._drag_root_start[0]
        dy = int(event.y_root) - self._drag_root_start[1]
        ox0, oy0 = self._start_offset
        self.set_offset((ox0 + dx, oy0 + dy))
        try:
            self.redraw()
        except Exception:
            pass
        return 'break'

    def _on_pan_end(self, event):
        self._dragging = False
        try:
            self.widget.configure(cursor='')
        except Exception:
            pass
        return 'break'


class ScrollZoomPan(_Base):
    """Zoom/pan handler for scrollregion-based canvases (xview/yview)."""
    def __init__(
        self,
        widget: tk.Canvas,
        get_zoom: Callable[[], float],
        set_zoom: Callable[[float], None],
        redraw: Callable[[], None],
        min_zoom: float = 0.1,
        max_zoom: float = 10.0,
        wheel_factor: float = 1.1,
    ):
        super().__init__(widget)
        self.get_zoom = get_zoom
        self.set_zoom = set_zoom
        self.redraw = redraw
        self.min_zoom = float(min_zoom)
        self.max_zoom = float(max_zoom)
        self.wheel_factor = float(wheel_factor)
        self._bind()

    def _bind(self):
        try:
            self.widget.bind('<MouseWheel>', self._on_wheel, add='+')
            self.widget.bind('<Button-3>', self._on_pan_start, add='+')
            self.widget.bind('<B3-Motion>', self._on_pan_move, add='+')
            self.widget.bind('<ButtonRelease-3>', self._on_pan_end, add='+')
        except Exception:
            pass

    def detach(self):
        for seq in ('<MouseWheel>', '<Button-3>', '<B3-Motion>', '<ButtonRelease-3>'):
            try:
                self.widget.unbind(seq)
            except Exception:
                pass

    def _on_wheel(self, event):
        if self._is_ctrl(event):
            return
        try:
            delta = int(getattr(event, 'delta', 0))
        except Exception:
            delta = 0
        if delta == 0:
            return
        old_zoom = float(self.get_zoom())
        factor = self.wheel_factor if delta > 0 else (1.0 / self.wheel_factor)
        new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        # Cursor relative position BEFORE zoom
        try:
            mouse_x = int(getattr(event, 'x_root')) - int(self.widget.winfo_rootx())
            mouse_y = int(getattr(event, 'y_root')) - int(self.widget.winfo_rooty())
        except Exception:
            try:
                mouse_x = int(getattr(event, 'x', self.widget.winfo_width() // 2))
                mouse_y = int(getattr(event, 'y', self.widget.winfo_height() // 2))
            except Exception:
                mouse_x = self.widget.winfo_width() // 2
                mouse_y = self.widget.winfo_height() // 2

        left_px_before = float(self.widget.canvasx(0))
        top_px_before = float(self.widget.canvasy(0))
        canvas_x_before = left_px_before + float(mouse_x)
        canvas_y_before = top_px_before + float(mouse_y)
        zoom_ratio = new_zoom / (old_zoom if old_zoom != 0 else 1e-6)

        self.set_zoom(new_zoom)
        try:
            self.redraw()
        except Exception:
            pass
        try:
            self.widget.update_idletasks()
        except Exception:
            pass

        canvas_x_after = canvas_x_before * zoom_ratio
        canvas_y_after = canvas_y_before * zoom_ratio

        bbox = self.widget.bbox('all')
        if bbox:
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            # Keep current scroll position; only log new zoom & bbox size
            try:
                cur_x = self.widget.xview()[0]
                cur_y = self.widget.yview()[0]
            except Exception:
                cur_x, cur_y = 0.0, 0.0

    def _on_pan_start(self, event):
        self._dragging = True
        self._drag_root_start = (int(event.x_root), int(event.y_root))
        # store pixel scroll offsets
        try:
            bbox = self.widget.bbox('all')
            if bbox:
                width = float(bbox[2] - bbox[0])
                height = float(bbox[3] - bbox[1])
                x_frac = self.widget.xview()[0]
                y_frac = self.widget.yview()[0]
                self._scroll_start_px = (x_frac * width, y_frac * height)
            else:
                self._scroll_start_px = (0.0, 0.0)
        except Exception:
            self._scroll_start_px = (0.0, 0.0)
        try:
            self.widget.configure(cursor='fleur')
        except Exception:
            pass
        return 'break'

    def _on_pan_move(self, event):
        if not self._dragging or not self._scroll_start_px:
            return
        dx = int(event.x_root) - self._drag_root_start[0]
        dy = int(event.y_root) - self._drag_root_start[1]
        start_x_px, start_y_px = self._scroll_start_px
        bbox = self.widget.bbox('all')
        if not bbox:
            return 'break'
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        new_x_px = start_x_px - dx
        new_y_px = start_y_px - dy
        new_x_frac = 0.0 if width <= 0 else max(0.0, min(1.0, new_x_px / width))
        new_y_frac = 0.0 if height <= 0 else max(0.0, min(1.0, new_y_px / height))
        try:
            self.widget.xview_moveto(new_x_frac)
            self.widget.yview_moveto(new_y_frac)
        except Exception:
            pass
        return 'break'

    def _on_pan_end(self, event):
        self._dragging = False
        try:
            self.widget.configure(cursor='')
        except Exception:
            pass
        return 'break'


__all__ = ['OffsetZoomPan', 'ScrollZoomPan']
