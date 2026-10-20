"""
Detection tab (v2).

Fringe centre lines are found by fringe_detection.ridge_detector, which follows
the local fringe direction and spacing instead of using fixed horizontal line
kernels. Heavy work runs in a background thread and is cached in stages, so
moving a slider only redoes the part of the analysis that slider affects:

  Max Fringe Angle, Smoothing Along Fringes -> full analysis (a few seconds)
  Sensitivity, Gap Bridging, Minimum Fringe Length, Noise Rejection
                                             -> clean-up only (about a second)

The viewer only draws the part of the image that is on screen, so zooming and
panning stay fast on the 4096 grid.

Analysis region: left-drag on the image to draw a box. Only fringes inside the
box are traced (and saved), and Auto-Optimize only scores inside it. The box is
stored in native pixel coordinates, so it survives a grid switch, and it is kept
when the next shot is loaded. Clear Region goes back to the whole image.
"""
import os
import queue
import threading
import traceback
import types
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from fringe_detection.ui_helpers import make_slider_row
from fringe_detection.ridge_detector import RidgeDetector, DEFAULTS, auto_optimize


class DetectionTabFrame(ttk.Frame):
    """Detection tab: detect fringe centre lines and show them over the shot."""

    # slider label -> (detector parameter, min, max, is_int, fmt, default, tooltip)
    SLIDERS = [
        ('Sensitivity', 'sensitivity', 0.0, 1.0, False, '{:.2f}', DEFAULTS['sensitivity'],
         'How faint a fringe can be and still be traced. Raise it to pick up more fringes near '
         'the target; lower it if noise starts being traced.'),
        ('Smoothing Along Fringes (px)', 'smooth_len', 0.0, 15.0, False, '{:.1f}', DEFAULTS['smooth_len'],
         'Averages the image along each fringe (never across it) to suppress speckle. Higher '
         'gives straighter, less broken lines in noisy areas; too high can make lines drift '
         'across dense, curved fringes. Automatically reduced where fringes are close together. '
         'Re-runs the full analysis.'),
        ('Max Fringe Angle°', 'max_angle', 10.0, 85.0, False, '{:.0f}', DEFAULTS['max_angle'],
         'Steepest fringe tilt (from horizontal) that is kept. Anything steeper, such as the '
         'vertical streaks at the target edge, is filtered out. Re-runs the full analysis.'),
        ('Gap Bridging (px)', 'gap_bridge', 0, 40, True, None, DEFAULTS['gap_bridge'],
         'Largest break in a fringe that is joined automatically. A gap is only bridged if the '
         'two ends line up along the fringe direction and the bridge does not cross a dark fringe.'),
        ('Minimum Fringe Length (px)', 'min_length', 5, 200, True, None, DEFAULTS['min_length'],
         'Pieces shorter than this are deleted. Lengths are in native (2048) pixels on either grid.'),
        ('Noise Rejection', 'mask_level', 0.0, 1.0, False, '{:.2f}', DEFAULTS['mask_level'],
         'How hard short pieces are rejected where the fringe pattern is weak or disordered '
         '(shadows, the dark band above the plume). Long fringes are always kept. 0 keeps everything.'),
    ]
    STAGE_A_KEYS = ('smooth_len', 'max_angle')

    def __init__(self, master, status_callback=None):
        super().__init__(master)
        self._status_cb = status_callback or (lambda txt: None)
        # Set by the app: callable(binary_uint8, shot_float, saved_path, name) -> bool
        self.send_to_editor_cb = None

        # Data/state
        self.src_img = None          # raw image as float32 (native grid)
        self.enh_img = None          # uint8 display image on the working grid (for Editor compatibility)
        self._src_path = None
        self._grid_factor = 1
        self._det = None             # RidgeDetector for the current image
        self._binary_mask = None     # uint8, 0 = fringe, 255 = background (what gets saved)
        self._traced = None          # bool, True = fringe pixel, working grid
        self._disp_native = None     # uint8 display image, native grid
        self._flat_native = None     # uint8 flattened fringe image, native grid
        self.robust_range = tk.BooleanVar(value=False)
        self.show_bg_estimate = tk.BooleanVar(value=False)

        # Worker / scheduling
        self._q = queue.Queue()
        self._busy = False
        self._pending = None
        self._debounce_id = None
        self._job_id = 0
        self._suspend_param_cb = False

        # Viewer state: zoom = screen px per working-grid px; (ox, oy) = image coords at canvas (0, 0)
        self._zoom = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._photo = None
        self._img_item = None
        self._pan_start = None
        self._render_id = None

        # Analysis region (native pixel coords x0, y0, x1, y1) and box-drawing state
        self._roi = None
        self._roi_drag = None
        self._roi_drag_item = None
        self._roi_item = None

        self._build_ui()
        self._attach_handlers()
        self.after(50, self._poll_queue)

    # ------------------------------------------------------------------ UI --
    def _make_help_icon(self, parent_frame, tooltip_text, side='left'):
        try:
            bg = self.cget('background')
        except Exception:
            bg = '#f0f0f0'
        c = tk.Canvas(parent_frame, width=18, height=18, highlightthickness=0, bg=bg)
        c.create_oval(2, 2, 16, 16, outline='#666', width=1)
        c.create_text(9, 9, text='?', font=('Segoe UI', 9))
        c.pack(side='right', padx=(0, 6))
        self._attach_tooltip(c, tooltip_text, side=side)
        return c

    def _build_ui(self):
        # Left control panel container (scrollable)
        ctrl_container = ttk.Frame(self)
        ctrl_container.pack(side='left', fill='y', padx=8, pady=8)
        ctrl_container.config(width=280)
        ctrl_container.pack_propagate(False)
        try:
            bg_color = self.cget('background') or '#f0f0f0'
        except Exception:
            bg_color = '#f0f0f0'
        canvas = tk.Canvas(ctrl_container, highlightthickness=0, bg=bg_color)
        scrollbar = ttk.Scrollbar(ctrl_container, orient='vertical', command=canvas.yview)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        canvas.configure(yscrollcommand=scrollbar.set)
        ctrl = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=ctrl, anchor='nw')

        def check_scrollbar():
            bbox = canvas.bbox('all')
            if not bbox:
                return
            if bbox[3] > canvas.winfo_height():
                scrollbar.pack(side='right', fill='y')
            else:
                scrollbar.pack_forget()

        ctrl.bind('<Configure>', lambda e: (canvas.configure(scrollregion=canvas.bbox('all')), check_scrollbar()))
        canvas.bind('<Configure>', lambda e: (canvas.itemconfig(canvas_window, width=e.width), check_scrollbar()))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            return 'break'

        def _recursive_bind(widget):
            widget.bind('<MouseWheel>', _on_mousewheel)
            for child in widget.winfo_children():
                _recursive_bind(child)

        canvas.bind('<MouseWheel>', _on_mousewheel)
        self.after_idle(lambda: _recursive_bind(ctrl))

        left_title = ttk.Frame(ctrl)
        left_title.pack(anchor='w', fill='x')
        ttk.Label(left_title, text='Fringe Detection', font=('Segoe UI', 10, 'bold')).pack(side='left')

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(anchor='w', pady=4)
        ttk.Button(btn_row, text='Load Image', command=self.load_image_dialog).pack(side='left')
        ttk.Button(btn_row, text='Save Fringes as Binary', command=self.save_result).pack(side='left', padx=(6, 0))

        send_row = ttk.Frame(ctrl)
        send_row.pack(anchor='w', fill='x', pady=(2, 0))
        ttk.Button(send_row, text='Send to Editor', command=self.send_to_editor).pack(side='left', fill='x', expand=True)
        self._make_help_icon(send_row,
                             'Saves the current fringes to EditedImages (named after the shot, never '
                             'overwriting an older file) and opens them in the Editor with the shot image '
                             'underneath, already aligned.')

        auto_row = ttk.Frame(ctrl)
        auto_row.pack(anchor='w', fill='x', pady=(4, 2))
        self.btn_auto = ttk.Button(auto_row, text='Auto-Optimize', command=self.auto_optimize)
        self.btn_auto.pack(side='left', fill='x', expand=True)
        ttk.Button(auto_row, text='Fit', width=4, command=self.fit_view).pack(side='left', padx=(6, 0))
        self._make_help_icon(auto_row,
                             'Tries 20 combinations of Sensitivity, Smoothing and Noise Rejection on this '
                             'image and keeps the one giving the longest, cleanest fringes: fewest broken '
                             'ends, stray pieces and lines squeezed between neighbouring fringes. Takes '
                             'about a minute. The sliders are set to the result, so you can fine-tune after.')

        roi_row = ttk.Frame(ctrl)
        roi_row.pack(anchor='w', fill='x', pady=(2, 2))
        self.btn_clear_roi = ttk.Button(roi_row, text='Clear Region', command=self.clear_region)
        self.btn_clear_roi.pack(side='left')
        self.roi_label = ttk.Label(roi_row, text='Region: whole image')
        self.roi_label.pack(side='left', padx=(6, 0))
        self._make_help_icon(roi_row,
                             'Left-click and drag on the image to draw an analysis region. Only fringes '
                             'inside the box are traced and saved, and Auto-Optimize only scores inside it. '
                             'Drag again to replace it. The box stays when you load the next shot. '
                             'Clear Region goes back to the whole image.')

        ttk.Checkbutton(ctrl, text='Brighten Image', variable=self.robust_range,
                        command=self._on_display_change).pack(anchor='w', padx=6)

        self._slider_meta = {}
        self._vars = {}
        self.bg_opacity = tk.DoubleVar(value=1.0)
        s_bg = make_slider_row(ctrl, 'Image Opacity', self.bg_opacity, 0.0, 1.0, is_int=False, fmt='{:.2f}',
                               command=lambda _=None: self._on_display_change())
        self._slider_meta['Image Opacity'] = {'scale': s_bg, 'var': self.bg_opacity, 'is_int': False, 'frm': 0.0, 'to': 1.0}
        self._make_help_icon(ctrl.winfo_children()[-1],
                             'Adjusts the transparency of the raw interferometry image. Increase for higher contrast')

        self.status = ttk.Label(ctrl, text='Ready', wraplength=240)
        self.status.pack(anchor='w', pady=(0, 6))
        self.progress = ttk.Progressbar(ctrl, mode='determinate', length=200, maximum=100)
        self.progress.pack(anchor='w', fill='x', pady=(0, 4))

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(ctrl, text='Detection Settings', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(4, 2))

        for (label, key, frm, to, is_int, fmt, default, tip) in self.SLIDERS:
            var = tk.IntVar(value=int(default)) if is_int else tk.DoubleVar(value=float(default))
            sc = make_slider_row(ctrl, label, var, frm, to, is_int=is_int, fmt=fmt,
                                 command=lambda _=None, k=key: self.on_param_change(k))
            self._slider_meta[label] = {'scale': sc, 'var': var, 'is_int': is_int, 'frm': frm, 'to': to}
            self._vars[key] = var
            self._make_help_icon(ctrl.winfo_children()[-1], tip)

        ttk.Checkbutton(ctrl, text='Show Flattened Image', variable=self.show_bg_estimate,
                        command=self._on_display_change).pack(anchor='w', pady=(6, 4), padx=6)

        ttk.Button(ctrl, text='Edit Sliders', command=self.open_slider_ranges).pack(anchor='w', pady=(6, 2))
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        ttk.Frame(ctrl, height=100).pack(fill='x')

        # Viewer
        img_frame = ttk.Frame(self)
        img_frame.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        img_frame.pack_propagate(False)
        self.fringe_canvas = tk.Canvas(img_frame, bg='black', highlightthickness=0)
        self.fringe_canvas.pack(fill='both', expand=True)

    def _attach_handlers(self):
        c = self.fringe_canvas
        c.bind('<MouseWheel>', self._on_wheel)
        c.bind('<Button-4>', lambda e: self._on_wheel(types.SimpleNamespace(delta=120, x=e.x, y=e.y)))
        c.bind('<Button-5>', lambda e: self._on_wheel(types.SimpleNamespace(delta=-120, x=e.x, y=e.y)))
        c.bind('<Button-3>', self._on_pan_start)
        c.bind('<B3-Motion>', self._on_pan_move)
        c.bind('<ButtonRelease-3>', self._on_pan_end)
        c.bind('<Button-1>', self._on_roi_start)
        c.bind('<B1-Motion>', self._on_roi_move)
        c.bind('<ButtonRelease-1>', self._on_roi_end)
        c.bind('<Configure>', lambda e: self._schedule_view())

    # ------------------------------------------------------------ helpers --
    def set_status(self, txt):
        try:
            self.status.config(text=txt)
        except Exception:
            pass
        try:
            self._status_cb(txt)
        except Exception:
            pass

    def _attach_tooltip(self, widget, text, side='right'):
        tip = {'win': None}

        def show_tip(_e=None):
            if tip['win'] is not None:
                return
            win = tk.Toplevel(widget)
            tip['win'] = win
            try:
                win.wm_overrideredirect(True)
            except Exception:
                pass
            frame = ttk.Frame(win, borderwidth=1, relief='solid')
            frame.pack()
            ttk.Label(frame, text=text, justify='left', padding=6, wraplength=220).pack()
            try:
                win.update_idletasks()
                wrx = widget.winfo_rootx(); wry = widget.winfo_rooty()
                ww = widget.winfo_width(); wh = widget.winfo_height()
                win_w = win.winfo_width() or win.winfo_reqwidth()
                win_h = win.winfo_height() or win.winfo_reqheight()
                x = int(wrx - 12 - win_w) if str(side).lower() == 'left' else int(wrx + ww + 12)
                y = int(wry + wh * 0.5 - win_h * 0.5)
                win.wm_geometry(f'+{max(0, x)}+{max(0, y)}')
            except Exception:
                pass

        def hide_tip(_e=None):
            w = tip.get('win')
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass
                tip['win'] = None

        try:
            widget.bind('<Enter>', show_tip)
            widget.bind('<Leave>', hide_tip)
        except Exception:
            pass

    def _params(self):
        """Snapshot slider values (main thread only)."""
        p = {}
        for (label, key, frm, to, is_int, fmt, default, tip) in self.SLIDERS:
            v = self._vars[key].get()
            p[key] = int(round(v)) if is_int else float(v)
        p['roi'] = self._roi
        return p

    def _set_params(self, p):
        self._suspend_param_cb = True
        try:
            for (label, key, frm, to, is_int, fmt, default, tip) in self.SLIDERS:
                if key in p:
                    v = p[key]
                    meta = self._slider_meta[label]
                    # widen the slider range if the optimiser went outside it
                    if v < meta['frm'] or v > meta['to']:
                        meta['frm'] = min(meta['frm'], v); meta['to'] = max(meta['to'], v)
                        meta['scale'].config(from_=meta['frm'], to=meta['to'])
                    self._vars[key].set(int(round(v)) if is_int else float(v))
        finally:
            self._suspend_param_cb = False

    # ------------------------------------------------------------ image IO --
    def _initial_dir(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        edited_dir = os.path.join(os.path.dirname(base_dir), 'EditedImages')
        return edited_dir if os.path.isdir(edited_dir) else os.path.dirname(base_dir)

    def load_image_dialog(self):
        try:
            self.lift()
        except Exception:
            pass
        p = filedialog.askopenfilename(parent=self, title='Select image', initialdir=self._initial_dir(),
                                       filetypes=[('Images', ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'))])
        if not p:
            return
        self.load_image(p)

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror('Load error', f'Could not read {path}')
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
        self.src_img = img.astype(np.float32)
        self._src_path = path
        # keep the analysis region for the next shot if it still fits inside it
        if self._roi is not None:
            h, w = self.src_img.shape[:2]
            x0, y0, x1, y1 = self._roi
            if x1 > w or y1 > h:
                self._roi = None
        self._update_roi_label()
        self._det = RidgeDetector(self.src_img, self._params())
        self._traced = None
        self._binary_mask = None
        self._flat_native = None
        self._make_display_image()
        h, w = self.src_img.shape[:2]
        self.set_status(f'Loaded: {os.path.basename(path)} - {w}x{h}')
        self.after(60, self.fit_view)
        self.start_render_now()

    def _make_display_image(self):
        if self.src_img is None:
            return
        work = self.src_img
        if self.robust_range.get():
            lo, hi = np.percentile(work, (0.1, 99.9))
        else:
            lo, hi = float(work.min()), float(work.max())
        if hi <= lo:
            self._disp_native = np.zeros(work.shape, np.uint8)
        else:
            self._disp_native = np.clip((work - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        self._update_enh_img()

    def _update_enh_img(self):
        # Kept for compatibility with apply_editor_mask: a display image on the working grid
        if self._disp_native is None:
            self.enh_img = None
            return
        s = self._grid_factor
        if s == 1:
            self.enh_img = self._disp_native
        else:
            h, w = self._disp_native.shape
            self.enh_img = cv2.resize(self._disp_native, (w * s, h * s), interpolation=cv2.INTER_CUBIC)

    def set_grid_factor(self, factor):
        """Switch between native (factor=1) and 2x supersampled (factor=2) output.

        The analysis itself always runs on the native image; only the ridge
        centre-line step and the clean-up run on the chosen grid, so switching
        is quick and sub-pixel positions are kept on the 4096 grid.
        """
        factor = max(1, int(factor))
        if factor == self._grid_factor:
            return
        old = self._grid_factor
        self._grid_factor = factor
        self._zoom = max(0.02, min(64.0, self._zoom * old / float(factor)))
        self._ox *= factor / float(old)
        self._oy *= factor / float(old)
        self._binary_mask = None
        self._traced = None
        self._update_enh_img()
        self._update_roi_label()
        if self.src_img is not None:
            h, w = self.src_img.shape[:2]
            self.set_status(f'Grid {w * factor}x{h * factor} - re-detecting')
            self.start_render_now()
        else:
            self.set_status(f'Grid set to {2048 * factor}px - load an image')
        self._schedule_view()

    def reload_source(self):
        if self._src_path:
            self.load_image(self._src_path)

    def save_result(self):
        if self._binary_mask is None:
            messagebox.showinfo('No result', 'No result to save')
            return
        edited_dir = self._initial_dir()
        try:
            os.makedirs(edited_dir, exist_ok=True)
        except Exception:
            pass
        p = filedialog.asksaveasfilename(initialdir=edited_dir, defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p:
            return
        try:
            if not cv2.imwrite(p, self._binary_mask):
                raise RuntimeError('cv2.imwrite returned False')
            self.set_status(f'Saved: {os.path.basename(p)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def _auto_save_path(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        edited_dir = os.path.join(os.path.dirname(base_dir), 'EditedImages')
        os.makedirs(edited_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self._src_path or 'shot'))[0]
        stem = f'{stem}_fringes_{2048 * self._grid_factor}'
        path = os.path.join(edited_dir, stem + '.png')
        k = 2
        while os.path.exists(path):
            path = os.path.join(edited_dir, f'{stem}_{k}.png')
            k += 1
        return path

    def send_to_editor(self):
        if self._binary_mask is None or self.src_img is None:
            messagebox.showinfo('Send to Editor', 'Detect fringes first (load an image)')
            return
        if self._busy:
            messagebox.showinfo('Send to Editor', 'Detection is still updating; try again in a moment')
            return
        try:
            path = self._auto_save_path()
            if not cv2.imwrite(path, self._binary_mask):
                raise RuntimeError('cv2.imwrite returned False')
        except Exception as e:
            messagebox.showerror('Save error', str(e))
            return
        sent = False
        if self.send_to_editor_cb is not None:
            try:
                sent = self.send_to_editor_cb(self._binary_mask, self.src_img, path,
                                              os.path.basename(self._src_path or 'shot'))
            except Exception:
                traceback.print_exc()
                sent = False
        if sent is False:
            self.set_status(f'Saved {os.path.basename(path)} (not sent to the Editor)')
        else:
            self.set_status(f'Saved {os.path.basename(path)} and sent to the Editor')

    # ------------------------------------------------- slider range dialog --
    def open_slider_ranges(self):
        dlg = tk.Toplevel(self)
        dlg.title('Edit slider ranges')
        dlg.transient(self)
        dlg.grab_set()
        rows = []
        for i, (name, meta) in enumerate(self._slider_meta.items()):
            ttk.Label(dlg, text=name).grid(row=i, column=0, sticky='w', padx=6, pady=4)
            frm_val = tk.StringVar(value=str(meta.get('frm', '')))
            to_val = tk.StringVar(value=str(meta.get('to', '')))
            ttk.Entry(dlg, textvariable=frm_val, width=8).grid(row=i, column=1, padx=4, pady=4)
            ttk.Entry(dlg, textvariable=to_val, width=8).grid(row=i, column=2, padx=4, pady=4)
            rows.append((name, meta, frm_val, to_val))

        def on_apply():
            for name, meta, frm_var, to_var in rows:
                try:
                    if meta['is_int']:
                        new_frm = int(float(frm_var.get())); new_to = int(float(to_var.get()))
                    else:
                        new_frm = float(frm_var.get()); new_to = float(to_var.get())
                except Exception:
                    messagebox.showerror('Invalid value', f'Invalid numeric value for {name}')
                    return
                if new_to <= new_frm:
                    messagebox.showerror('Invalid range', f'Max must be > Min for {name}')
                    return
                meta['frm'] = new_frm; meta['to'] = new_to
                scale = meta.get('scale')
                if scale is not None:
                    try:
                        scale.config(from_=new_frm, to=new_to)
                        var = meta.get('var')
                        if var is not None:
                            v = var.get()
                            if v < new_frm:
                                var.set(new_frm)
                            elif v > new_to:
                                var.set(new_to)
                    except Exception:
                        pass
            dlg.destroy()

        btn_fr = ttk.Frame(dlg)
        btn_fr.grid(row=len(rows), column=0, columnspan=3, pady=8)
        ttk.Button(btn_fr, text='Apply', command=on_apply).pack(side='left', padx=6)
        ttk.Button(btn_fr, text='Cancel', command=dlg.destroy).pack(side='left', padx=6)

    # ---------------------------------------------------------- scheduling --
    def on_param_change(self, key=None):
        if self._suspend_param_cb:
            return
        # Debounce: ttk.Scale fires continuously while dragging. Wait until the
        # slider has been still briefly; stage-A sliders wait a little longer.
        if self._debounce_id is not None:
            try:
                self.after_cancel(self._debounce_id)
            except Exception:
                pass
        delay = 450 if key in self.STAGE_A_KEYS else 200
        self._debounce_id = self.after(delay, self.start_render_now)

    def start_render_now(self):
        self._debounce_id = None
        if self.src_img is None or self._det is None:
            self.set_status('No image loaded')
            return
        job = ('detect', self._params(), self._grid_factor)
        if self._busy:
            self._pending = job
            self.set_status('Updating (queued)...')
            return
        self._launch(job)

    def _launch(self, job):
        self._busy = True
        self._job_id += 1
        jid = self._job_id
        try:
            self.progress.config(mode='indeterminate')
            self.progress.start(12)
        except Exception:
            pass
        try:
            self.btn_auto.state(['disabled'])
        except Exception:
            pass
        threading.Thread(target=self._worker, args=(jid, job), daemon=True).start()

    def _worker(self, jid, job):
        """Runs off the main thread. Never touches Tk; talks through the queue."""
        kind, params, up = job
        det = self._det
        try:
            prog = lambda msg: self._q.put(('status', jid, msg))
            if kind == 'detect':
                det.p.update(params)
                b = det.detect(up=up, progress=prog)
                self._q.put(('result', jid, (b, up, None)))
            elif kind == 'auto':
                best, b, report = auto_optimize(det, up=up, base=params, progress=prog)
                self._q.put(('result', jid, (b, up, (best, report))))
        except Exception as e:
            self._q.put(('error', jid, '{}: {}\n\n{}'.format(type(e).__name__, e, traceback.format_exc())))
        finally:
            self._q.put(('done', jid, None))

    def _poll_queue(self):
        try:
            while True:
                kind, jid, payload = self._q.get_nowait()
                if kind == 'status':
                    self.set_status(payload)
                elif kind == 'result':
                    b, up, auto = payload
                    if auto is not None:
                        best, report = auto
                        self._set_params(best)
                    if up == self._grid_factor:
                        self._set_result(b)
                    if auto is not None:
                        self.set_status(report)
                elif kind == 'error':
                    traceback_text = payload
                    print(traceback_text)
                    messagebox.showerror('Detection error', traceback_text.split('\n\n')[0])
                    self.set_status('Error - see message')
                elif kind == 'done':
                    self._busy = False
                    try:
                        self.progress.stop()
                        self.progress.config(mode='determinate')
                        self.progress['value'] = 0
                    except Exception:
                        pass
                    try:
                        self.btn_auto.state(['!disabled'])
                    except Exception:
                        pass
                    nxt = self._pending
                    self._pending = None
                    if nxt is not None:
                        # always re-read the sliders: they may have moved again
                        if nxt[0] == 'detect':
                            nxt = ('detect', self._params(), self._grid_factor)
                        self._launch(nxt)
        except queue.Empty:
            pass
        except Exception:
            traceback.print_exc()
        try:
            self.after(50, self._poll_queue)
        except Exception:
            pass

    def _set_result(self, b):
        self._traced = b
        self._binary_mask = np.where(b, 0, 255).astype(np.uint8)
        if self._det is not None and self._det._A is not None:
            fl = self._det._A['flat']
            self._flat_native = np.clip(fl * 60 + 128, 0, 255).astype(np.uint8)
        n = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)[0] - 1
        t = sum(v for k, v in self._det.timings.items()) if self._det else 0
        self.set_status(f'Rendered - {n} fringe segments')
        self._schedule_view()

    # ----------------------------------------------------- analysis region --
    def _canvas_to_native(self, cx, cy):
        s = self._grid_factor
        return (self._ox + cx / self._zoom) / s, (self._oy + cy / self._zoom) / s

    def _native_to_canvas(self, nx, ny):
        s = self._grid_factor
        return (nx * s - self._ox) * self._zoom, (ny * s - self._oy) * self._zoom

    def _on_roi_start(self, event):
        if self.src_img is None:
            return
        self._roi_drag = (event.x, event.y)
        if self._roi_drag_item is not None:
            self.fringe_canvas.delete(self._roi_drag_item)
        self._roi_drag_item = self.fringe_canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline='#ffd400', width=2, dash=(6, 4))

    def _on_roi_move(self, event):
        if self._roi_drag is None or self._roi_drag_item is None:
            return
        x0, y0 = self._roi_drag
        self.fringe_canvas.coords(self._roi_drag_item, x0, y0, event.x, event.y)

    def _on_roi_end(self, event):
        if self._roi_drag is None:
            return
        x0, y0 = self._roi_drag
        self._roi_drag = None
        if self._roi_drag_item is not None:
            self.fringe_canvas.delete(self._roi_drag_item)
            self._roi_drag_item = None
        if abs(event.x - x0) < 6 or abs(event.y - y0) < 6:
            self.set_status('Drag to draw an analysis region (a click does nothing)')
            return
        h, w = self.src_img.shape[:2]
        ax, ay = self._canvas_to_native(x0, y0)
        bx, by = self._canvas_to_native(event.x, event.y)
        nx0, nx1 = sorted((float(np.clip(ax, 0, w)), float(np.clip(bx, 0, w))))
        ny0, ny1 = sorted((float(np.clip(ay, 0, h)), float(np.clip(by, 0, h))))
        # snap to the image border when the drag ends within a few screen pixels of it
        snap = 4.0 / (self._zoom * self._grid_factor)
        nx0 = 0.0 if nx0 < snap else nx0
        ny0 = 0.0 if ny0 < snap else ny0
        nx1 = float(w) if nx1 > w - snap else nx1
        ny1 = float(h) if ny1 > h - snap else ny1
        if nx1 - nx0 < 8 or ny1 - ny0 < 8:
            self.set_status('Analysis region must lie on the image')
            return
        self.set_region((nx0, ny0, nx1, ny1))

    def set_region(self, roi):
        """Set the analysis region (native pixel coords x0, y0, x1, y1), or None."""
        self._roi = None if roi is None else tuple(float(v) for v in roi)
        self._update_roi_label()
        self._schedule_view()
        if self.src_img is not None:
            self.start_render_now()

    def clear_region(self):
        self.set_region(None)

    def _update_roi_label(self):
        try:
            if self._roi is None:
                self.roi_label.config(text='Region: whole image')
            else:
                x0, y0, x1, y1 = self._roi
                s = self._grid_factor
                self.roi_label.config(text='Region: {}x{} px'.format(int(round((x1 - x0) * s)), int(round((y1 - y0) * s))))
        except Exception:
            pass

    # --------------------------------------------------------- auto-optimize --
    def auto_optimize(self):
        if self.src_img is None or self._det is None:
            messagebox.showinfo('Auto-Optimize', 'Load an image first')
            return
        if self._busy:
            self._pending = ('auto', self._params(), self._grid_factor)
            self.set_status('Auto-optimize queued...')
            return
        self._launch(('auto', self._params(), self._grid_factor))

    # ---------------------------------------------------- editor integration --
    def update_with_overlay(self, illum_bgr, overlay_bgr):
        self._schedule_view()

    def apply_editor_mask(self, mask):
        """Apply externally edited binary mask."""
        try:
            if mask is None:
                return
            if self.src_img is not None:
                h, w = self.src_img.shape[:2]
                gh, gw = h * self._grid_factor, w * self._grid_factor
                if mask.shape[:2] != (gh, gw):
                    # Never silently resize a binary mask: downsampling fuses
                    # neighbouring fringes. Refuse and say so.
                    messagebox.showerror(
                        'Grid mismatch',
                        'The edited mask is {}x{} but detection is running at {}x{}.\n\n'
                        'Set both tabs to the same grid before applying.'.format(mask.shape[1], mask.shape[0], gw, gh))
                    return
            m = mask.copy()
            self._binary_mask = np.where(m < 128, 0, 255).astype(np.uint8)
            self._traced = self._binary_mask == 0
            self._schedule_view()
            self.set_status('Applied edited mask from Editor')
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------- viewer --
    def _on_display_change(self):
        if self.src_img is not None and self._disp_native is not None:
            self._make_display_image()
        self._schedule_view()

    def _image_size(self):
        if self.src_img is None:
            return None
        h, w = self.src_img.shape[:2]
        return w * self._grid_factor, h * self._grid_factor

    def fit_view(self, force=True):
        sz = self._image_size()
        if sz is None:
            return
        iw, ih = sz
        cw = self.fringe_canvas.winfo_width(); ch = self.fringe_canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            if force:
                self.after(120, lambda: self.fit_view(force=False))
            return
        z = min(cw / float(iw), ch / float(ih))
        self._zoom = max(0.02, min(64.0, z))
        self._ox = (iw - cw / self._zoom) / 2.0
        self._oy = (ih - ch / self._zoom) / 2.0
        self._schedule_view()

    def _on_wheel(self, event):
        if self.src_img is None:
            return
        step = 1.15 if event.delta > 0 else 1 / 1.15
        nz = max(0.02, min(64.0, self._zoom * step))
        # keep the image point under the cursor fixed
        ix = self._ox + event.x / self._zoom
        iy = self._oy + event.y / self._zoom
        self._zoom = nz
        self._ox = ix - event.x / nz
        self._oy = iy - event.y / nz
        self._schedule_view(fast=True)

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y, self._ox, self._oy)
        try:
            self.fringe_canvas.configure(cursor='fleur')
        except Exception:
            pass
        return 'break'

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        x0, y0, ox, oy = self._pan_start
        self._ox = ox - (event.x - x0) / self._zoom
        self._oy = oy - (event.y - y0) / self._zoom
        self._schedule_view(fast=True)
        return 'break'

    def _on_pan_end(self, event):
        self._pan_start = None
        try:
            self.fringe_canvas.configure(cursor='')
        except Exception:
            pass
        return 'break'

    def _schedule_view(self, fast=False):
        if self._render_id is not None:
            return
        self._render_id = self.after(1 if fast else 10, self._render_view)

    def _render_view(self):
        """Draw only the visible part of the image at screen resolution."""
        self._render_id = None
        c = self.fringe_canvas
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        if self._disp_native is None:
            return
        s = self._grid_factor
        base = self._flat_native if (self.show_bg_estimate.get() and self._flat_native is not None) else self._disp_native
        H, W = base.shape[0] * s, base.shape[1] * s
        z = self._zoom
        # visible window in working-grid coordinates
        x0 = max(0.0, self._ox); y0 = max(0.0, self._oy)
        x1 = min(float(W), self._ox + cw / z); y1 = min(float(H), self._oy + ch / z)
        out = np.zeros((ch, cw, 3), np.uint8)
        if x1 > x0 and y1 > y0:
            # destination rectangle on the canvas
            dx0 = int(round((x0 - self._ox) * z)); dy0 = int(round((y0 - self._oy) * z))
            dx1 = int(round((x1 - self._ox) * z)); dy1 = int(round((y1 - self._oy) * z))
            dw, dh = max(1, dx1 - dx0), max(1, dy1 - dy0)
            dx1, dy1 = min(cw, dx0 + dw), min(ch, dy0 + dh)
            dw, dh = dx1 - dx0, dy1 - dy0
            if dw > 0 and dh > 0:
                # background from the native image
                nx0, ny0 = int(x0 // s), int(y0 // s)
                nx1, ny1 = int(np.ceil(x1 / s)), int(np.ceil(y1 / s))
                crop = base[ny0:ny1, nx0:nx1]
                # sub-pixel alignment of the crop to the window
                cz = z * s
                M = np.array([[cz, 0, (nx0 * s - x0) * z], [0, cz, (ny0 * s - y0) * z]], np.float32)
                interp = cv2.INTER_AREA if cz < 1 else (cv2.INTER_NEAREST if cz >= 4 else cv2.INTER_LINEAR)
                if cz < 1:
                    bg = cv2.resize(crop, (max(1, int(round(crop.shape[1] * cz))), max(1, int(round(crop.shape[0] * cz)))),
                                    interpolation=cv2.INTER_AREA)
                    M2 = np.array([[1, 0, (nx0 * s - x0) * z], [0, 1, (ny0 * s - y0) * z]], np.float32)
                    bg = cv2.warpAffine(bg, M2, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                else:
                    bg = cv2.warpAffine(crop, M, (dw, dh), flags=interp, borderMode=cv2.BORDER_REPLICATE)
                op = float(self.bg_opacity.get())
                bgf = bg.astype(np.float32) * op + 255.0 * (1.0 - op)
                tile = cv2.cvtColor(np.clip(bgf, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                # fringe lines from the working-grid mask
                tr = self._traced
                if tr is not None and tr.shape == (H, W):
                    gx0, gy0 = int(x0), int(y0)
                    gx1, gy1 = int(np.ceil(x1)), int(np.ceil(y1))
                    m = tr[gy0:gy1, gx0:gx1].astype(np.uint8)
                    if z < 1:
                        # max-pool so 1px lines stay visible when zoomed out
                        mw = max(1, int(round(m.shape[1] * z))); mh = max(1, int(round(m.shape[0] * z)))
                        ms = cv2.resize(m * 255, (mw, mh), interpolation=cv2.INTER_AREA) > 0
                        M3 = np.array([[1, 0, (gx0 - x0) * z], [0, 1, (gy0 - y0) * z]], np.float32)
                        ms = cv2.warpAffine(ms.astype(np.uint8), M3, (dw, dh), flags=cv2.INTER_NEAREST) > 0
                    else:
                        M3 = np.array([[z, 0, (gx0 - x0) * z], [0, z, (gy0 - y0) * z]], np.float32)
                        ms = cv2.warpAffine(m, M3, (dw, dh), flags=cv2.INTER_NEAREST) > 0
                    tile[ms] = (0, 0, 255)
                out[dy0:dy1, dx0:dx1] = tile
        if self._roi is not None:
            # dim everything outside the analysis region
            rx0, ry0 = self._native_to_canvas(self._roi[0], self._roi[1])
            rx1, ry1 = self._native_to_canvas(self._roi[2], self._roi[3])
            inside = np.zeros((ch, cw), bool)
            ix0, iy0 = int(np.clip(round(rx0), 0, cw)), int(np.clip(round(ry0), 0, ch))
            ix1, iy1 = int(np.clip(round(rx1), 0, cw)), int(np.clip(round(ry1), 0, ch))
            inside[iy0:iy1, ix0:ix1] = True
            out[~inside] = (out[~inside] * 0.45).astype(np.uint8)
        img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(img)
        if self._img_item is None:
            self._img_item = c.create_image(0, 0, anchor='nw', image=self._photo)
        else:
            c.itemconfig(self._img_item, image=self._photo)
        if self._roi is not None:
            rx0, ry0 = self._native_to_canvas(self._roi[0], self._roi[1])
            rx1, ry1 = self._native_to_canvas(self._roi[2], self._roi[3])
            if self._roi_item is None:
                self._roi_item = c.create_rectangle(rx0, ry0, rx1, ry1, outline='#ffd400', width=2)
            else:
                c.coords(self._roi_item, rx0, ry0, rx1, ry1)
            c.tag_raise(self._roi_item)
        elif self._roi_item is not None:
            c.delete(self._roi_item)
            self._roi_item = None
        if self._roi_drag_item is not None:
            c.tag_raise(self._roi_drag_item)
