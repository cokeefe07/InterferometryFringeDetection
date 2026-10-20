import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


def to_photoimage_from_bgr_with_scale(bgr, scale=1.0, interpolation=Image.BILINEAR):
    """Convert a BGR/gray numpy array to a Tk PhotoImage (optional scaling).

    Note: Avoids content caching to ensure immediate visual updates when the
    underlying numpy array contents change between calls.

    Args:
        bgr: BGR or grayscale numpy array
        scale: Scale factor (default=1.0)
        interpolation: PIL interpolation mode (default=Image.BILINEAR)
    """
    if bgr is None:
        return ImageTk.PhotoImage(Image.new('RGB', (1, 1)))

    # Always create a fresh PIL Image to reflect current pixel data
    # Optimization: Use cv2.resize which is significantly faster than PIL.Image.resize
    if scale is None:
        scale = 1.0
    
    scale = float(scale)
    if scale == 1.0:
        if bgr.ndim == 2:
            img = Image.fromarray(bgr)
        else:
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return ImageTk.PhotoImage(img)

    new_w = max(1, int(bgr.shape[1] * scale))
    new_h = max(1, int(bgr.shape[0] * scale))

    # Select interpolation method for speed/quality trade-off
    # Large upscaling -> Nearest Neighbor (fastest, crisp pixels)
    # Downscaling -> Area (best quality) or Linear (fast)
    if scale >= 2.0:
        cv_interp = cv2.INTER_NEAREST
    elif scale < 1.0:
        cv_interp = cv2.INTER_AREA
    else:
        # Map PIL constants to OpenCV if possible
        if interpolation == Image.NEAREST:
            cv_interp = cv2.INTER_NEAREST
        elif interpolation == Image.BICUBIC:
            cv_interp = cv2.INTER_CUBIC
        else:
            cv_interp = cv2.INTER_LINEAR

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv_interp)

    if resized.ndim == 2:
        img = Image.fromarray(resized)
    else:
        img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    return ImageTk.PhotoImage(img)


def make_slider_row(parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None, command=None):
    """Create a labeled slider with a live value label; returns the Scale widget."""
    if command is None:
        # no-op default
        def command(_=None):
            return
    # label above
    ttk.Label(parent, text=label_text).pack(anchor='w')
    row = ttk.Frame(parent)
    row.pack(fill='x')
    scale = ttk.Scale(row, from_=frm, to=to, variable=var, command=command)
    scale.pack(side='left', fill='x', expand=True)
    # value label
    val_var = tk.StringVar()
    if fmt is None:
        fmt = "{}"

    def _update_val(*a):
        try:
            v = var.get()
            if is_int:
                val_var.set(f"{int(round(v))}")
            else:
                if isinstance(v, float) and abs(v - round(v)) < 1e-6:
                    val_var.set(f"{int(round(v))}")
                else:
                    try:
                        val_var.set(fmt.format(v))
                    except Exception:
                        val_var.set(str(v))
        except Exception:
            val_var.set('')

    _update_val()
    try:
        var.trace_add('write', lambda *a: _update_val())
    except Exception:
        var.trace('w', lambda *a: _update_val())

    lbl = ttk.Label(row, textvariable=val_var, width=6, anchor='e')
    lbl.pack(side='left', padx=(6, 0))
    return scale
