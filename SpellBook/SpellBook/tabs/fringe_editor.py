"""
Fringe Editor (v2).

Same file formats and the same editing features as the original Editor, laid
out like the Detection tab: a control panel on the left, the image on the right.

Layers (all on the square working grid, 2048 or 4096):
  shot binary       self.mask          uint8, 0 = fringe, 255 = background
  gray mask         self.gray_mask     uint8, 255 = masked (saved as 128 in the binary)
  reference binary  self.overlay_mask  uint8, 0 = fringe, drawn in cyan at self.overlay_pos
  shot image        self._shot_img     uint8 photo plate
  reference image   self._ref_img      uint8 photo plate, drawn at self.ref_pos

Every tool acts on the layer chosen under "Editing" (shot or reference binary),
except Mask pen, which always edits the gray mask of the shot.

Undo/redo stores only the rectangle of pixels an edit changed, and while you
draw only the part of the screen under the pen is recomposited.
"""
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

ORANGE = (255, 128, 0)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
F = np.float32
OUTSIDE = (51, 51, 51)      # canvas colour outside the working grid (was gray20)

TOOLS = [
    # key, label, shortcut, tooltip
    ('pen', 'Pen', 'P', 'Draw fringe pixels on the layer selected under Editing. Size is set below '
                        '(Ctrl + mouse wheel also changes it). Shortcut: P'),
    ('erase', 'Eraser', 'E', 'Remove fringe pixels from the layer selected under Editing. Shortcut: E'),
    ('line', 'Line', 'L', 'Straight 1 px line: click and drag, or click the start then the end. '
                          'With Snap to ends on, clicks jump to the nearest fringe end. Shortcut: L'),
    ('spline', 'Spline', 'S', 'Smooth 1 px curve through your clicks. Click to add points; finish with '
                              'Ctrl+click, double click or Enter; Esc cancels. Shortcut: S'),
    ('delete', 'Delete Fringe', 'D', 'Click a line to delete that whole connected piece. Shortcut: D'),
    ('mask', 'Mask Pen', 'M', 'Paint the gray mask (regions Magic should ignore). Hold Shift while '
                              'painting to remove mask. Shortcut: M'),
    ('crop', 'Crop', 'C', 'Drag a box: everything outside it is cleared on all layers. '
                          'Reset Crop brings it back. Shortcut: C'),
    ('ruler', 'Ruler', 'R', 'Drag to measure in pixels. Drag either end to adjust; Lock keeps it '
                            'horizontal or vertical. Shortcut: R'),
    ('check', 'Connection Check', 'K', 'Drag a line across fringes: every fringe it touches turns orange. '
                                       'If two fringes share a colour they are connected. Shortcut: K'),
]
SIZED_TOOLS = ('pen', 'erase', 'mask')
DEFAULT_SIZES = {'pen': 0.5, 'erase': 1.5, 'mask': 10.0}


def _attach_tooltip(widget, text, side='right'):
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
        ttk.Label(frame, text=text, justify='left', padding=6, wraplength=240).pack()
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

    widget.bind('<Enter>', show_tip, add='+')
    widget.bind('<Leave>', hide_tip, add='+')
    widget.bind('<ButtonPress>', hide_tip, add='+')


class FringeEditorFrame(tk.Frame):
    """Embeddable fringe editor as a tkinter Frame."""

    def __init__(self, master=None, on_apply=None, on_close=None):
        super().__init__(master)
        self._on_apply = on_apply
        self._on_close = on_close

        # ---- data -------------------------------------------------------- #
        self.grid_factor = 1
        self.grid = 2048 * self.grid_factor
        self.mask = np.full((self.grid, self.grid), 255, dtype=np.uint8)
        self.gray_mask = np.zeros((self.grid, self.grid), dtype=np.uint8)
        self.overlay_mask = None
        self.overlay_pos = (0, 0)
        self.overlay_active = False
        self._shot_img = None
        self._ref_img = None
        self.ref_pos = (0, 0)
        self._initial_ref_pos = (0, 0)
        self._initial_overlay_pos = (0, 0)
        self._crop_mask = np.full((self.grid, self.grid), 255, dtype=np.uint8)
        self._uncropped = {}
        self._dirty = False                 # unsaved edits on the shot binary

        # ---- undo / redo (patches) ---------------------------------------- #
        self._undo = []
        self._redo = []
        self._snap_before = None
        self.MAX_UNDO = 100

        # ---- caches ------------------------------------------------------- #
        self._labels_cache = None           # (labels, colors) of shot binary
        self._labels_dirty = True
        self._ends_cache = {}               # layer -> (ys, xs) of fringe ends
        self._ver = {}                      # layer -> version, bumped on every change
        self._pyr = {}                      # (layer, kind, level) -> (version, array)
        self.magic2_labels = set()

        # ---- view state --------------------------------------------------- #
        self._zoom = 1.0                    # screen px per grid px
        self._ox = 0.0                      # grid coords at canvas (0, 0)
        self._oy = 0.0
        self._fitted = False
        self._frame_rgb = None
        self._photo = None
        self._img_item = None
        self._render_id = None

        # ---- interaction state --------------------------------------------- #
        self._painting = False
        self._paint_layer = None
        self._last_paint_xy = None
        self._tool_points = []
        self._preview_items = []
        self._drag_start = None
        self._ruler_points = []
        self._ruler_items = []
        self._ruler_drag_idx = None
        self._cursor_item = None
        self._snap_item = None
        self._last_mouse = None
        self._pan_start = None
        self._moving_overlay = False
        self._sizes = dict(DEFAULT_SIZES)

        self._build_ui()
        self._bind_events()
        self._set_tool('pen')

    # ================================================================== UI ==
    def _help(self, parent, text, side='left'):
        try:
            bg = self.cget('background')
        except Exception:
            bg = '#f0f0f0'
        c = tk.Canvas(parent, width=18, height=18, highlightthickness=0, bg=bg)
        c.create_oval(2, 2, 16, 16, outline='#666', width=1)
        c.create_text(9, 9, text='?', font=('Segoe UI', 9))
        c.pack(side='right', padx=(0, 4))
        _attach_tooltip(c, text, side=side)
        return c

    def _section(self, parent, title, help_text):
        row = ttk.Frame(parent)
        row.pack(anchor='w', fill='x', pady=(8, 2))
        ttk.Label(row, text=title, font=('Segoe UI', 9, 'bold')).pack(side='left')
        self._help(row, help_text)
        return row

    def _slider(self, parent, label, var, frm, to, command, fmt='{:.2f}', help_text=None):
        row = ttk.Frame(parent)
        row.pack(anchor='w', fill='x')
        ttk.Label(row, text=label, width=11).pack(side='left')
        sc = ttk.Scale(row, from_=frm, to=to, variable=var, command=lambda _=None: command())
        sc.pack(side='left', fill='x', expand=True)
        val = ttk.Label(row, width=5, anchor='e')
        val.pack(side='left', padx=(4, 0))

        def upd(*_):
            try:
                val.config(text=fmt.format(var.get()))
            except Exception:
                pass
        upd()
        var.trace_add('write', upd)
        if help_text:
            self._help(row, help_text)
        return sc

    def _build_ui(self):
        # --- left control panel (scrollable) ---
        container = ttk.Frame(self)
        container.pack(side='left', fill='y', padx=8, pady=8)
        container.config(width=290)
        container.pack_propagate(False)
        try:
            bg_color = self.cget('background') or '#f0f0f0'
        except Exception:
            bg_color = '#f0f0f0'
        pc = tk.Canvas(container, highlightthickness=0, bg=bg_color)
        sb = ttk.Scrollbar(container, orient='vertical', command=pc.yview)
        pc.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        pc.configure(yscrollcommand=sb.set)
        ctrl = ttk.Frame(pc)
        win = pc.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>', lambda e: pc.configure(scrollregion=pc.bbox('all')))
        pc.bind('<Configure>', lambda e: pc.itemconfig(win, width=e.width))

        def _wheel(event):
            pc.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            return 'break'

        def _bind_wheel(w):
            w.bind('<MouseWheel>', _wheel, add='+')
            for ch in w.winfo_children():
                _bind_wheel(ch)
        self.after_idle(lambda: _bind_wheel(ctrl))
        self._panel = ctrl

        title = ttk.Frame(ctrl)
        title.pack(anchor='w', fill='x')
        ttk.Label(title, text='Editor', font=('Segoe UI', 10, 'bold')).pack(side='left')
        self._help(title, 'Right-drag: pan.  Mouse wheel: zoom.  Ctrl + wheel: pen size.\n'
                          'Ctrl + right-drag or arrow keys: move the reference image and binary.\n'
                          'Ctrl+Z: undo.  Ctrl+Y or Ctrl+Shift+Z: redo.\n'
                          'Tool shortcuts are listed on each tool button.')

        # --- Files ---
        self._section(ctrl, 'Files', 'Open loads a binary or an image onto the current grid. Binaries '
                                     'keep their gray mask (value 128). Save writes exactly what each '
                                     'item says, as a 0/255 PNG with the gray mask as 128.')
        frow = ttk.Frame(ctrl)
        frow.pack(anchor='w', fill='x', pady=2)
        mb_open = ttk.Menubutton(frow, text='Open')
        m_open = tk.Menu(mb_open, tearoff=False)
        m_open.add_command(label='Shot Binary...', command=self.open_binary)
        m_open.add_command(label='Reference Binary...', command=self.open_overlay_binary)
        m_open.add_separator()
        m_open.add_command(label='Shot Image...', command=self._load_shot)
        m_open.add_command(label='Reference Image...', command=self._load_reference)
        m_open.add_separator()
        m_open.add_command(label='Gray Mask...', command=self.open_gray_mask)
        mb_open['menu'] = m_open
        mb_open.pack(side='left')
        mb_save = ttk.Menubutton(frow, text='Save')
        m_save = tk.Menu(mb_save, tearoff=False)
        m_save.add_command(label='Shot Binary...', command=self.save_binary)
        m_save.add_command(label='Reference Binary...', command=self.save_reference_binary)
        m_save.add_command(label='Gray Mask...', command=self.save_gray_mask)
        mb_save['menu'] = m_save
        mb_save.pack(side='left', padx=(6, 0))
        self.btn_fit = ttk.Button(frow, text='Fit', width=4, command=self.fit_view)
        self.btn_fit.pack(side='left', padx=(6, 0))

        # --- Editing ---
        self._section(ctrl, 'Editing', 'Which binary the tools change: the shot binary (black) or '
                                       'the reference binary (cyan). Mask Pen always edits the gray '
                                       'mask of the shot.')
        self.edit_layer_var = tk.StringVar(value='shot')
        erow = ttk.Frame(ctrl)
        erow.pack(anchor='w', fill='x')
        ttk.Radiobutton(erow, text='Shot binary', value='shot', variable=self.edit_layer_var,
                        command=self._on_edit_layer).pack(side='left')
        ttk.Radiobutton(erow, text='Reference binary', value='ref', variable=self.edit_layer_var,
                        command=self._on_edit_layer).pack(side='left', padx=(8, 0))

        # --- Tools ---
        self._section(ctrl, 'Tools', 'Pick one tool. Hover a tool for what it does and its keyboard '
                                     'shortcut.')
        self.mode_var = tk.StringVar(value='pen')
        tgrid = ttk.Frame(ctrl)
        tgrid.pack(anchor='w', fill='x')
        self._tool_buttons = {}
        for i, (key, label, sc, tip) in enumerate(TOOLS):
            b = ttk.Radiobutton(tgrid, text=f'{label} ({sc})', value=key, variable=self.mode_var,
                                style='Toolbutton', command=lambda k=key: self._set_tool(k))
            b.grid(row=i // 2, column=i % 2, sticky='ew', padx=1, pady=1)
            _attach_tooltip(b, tip)
            self._tool_buttons[key] = b
        tgrid.columnconfigure(0, weight=1)
        tgrid.columnconfigure(1, weight=1)

        # --- Tool options (contextual) ---
        self._section(ctrl, 'Tool Options', 'Settings for the selected tool. Only the ones that apply '
                                            'are shown.')
        self._opts = ttk.Frame(ctrl)
        self._opts.pack(anchor='w', fill='x')
        self.brush_var = tk.DoubleVar(value=DEFAULT_SIZES['pen'])
        self._opt_size = ttk.Frame(self._opts)
        self._slider(self._opt_size, 'Size (px)', self.brush_var, 0.5, 60.0, self._on_size_change, fmt='{:.1f}',
                     help_text='Radius of the pen, eraser or mask pen in working-grid pixels. 0.5 draws a '
                               '1 px line. Ctrl + mouse wheel changes it while hovering the image.')
        self.snap_var = tk.BooleanVar(value=True)
        self._opt_snap = ttk.Frame(self._opts)
        ttk.Checkbutton(self._opt_snap, text='Snap to ends', variable=self.snap_var,
                        command=self._update_snap_marker).pack(side='left')
        self._help(self._opt_snap, 'Line and Spline clicks jump to the nearest fringe end (marked with a '
                                   'green circle), so a join is exactly connected.')
        self._opt_ruler = ttk.Frame(self._opts)
        ttk.Label(self._opt_ruler, text='Lock:').pack(side='left')
        self.ruler_lock_var = tk.StringVar(value='None')
        ttk.Combobox(self._opt_ruler, textvariable=self.ruler_lock_var, values=['None', 'Horizontal', 'Vertical'],
                     width=10, state='readonly').pack(side='left', padx=(4, 0))
        self._help(self._opt_ruler, 'Keep the ruler horizontal or vertical while you drag.')
        self._opt_hint = ttk.Label(self._opts, text='', wraplength=250, foreground='#444')

        # --- Fix-ups ---
        self._section(ctrl, 'Fix-ups', 'Automatic repairs on the binary selected under Editing.')
        lrow = ttk.Frame(ctrl)
        lrow.pack(anchor='w', fill='x', pady=1)
        ttk.Button(lrow, text='Link Endpoints', command=self._link_endpoints).pack(side='left')
        ttk.Label(lrow, text='Angle°').pack(side='left', padx=(6, 2))
        self.angle_deg_var = tk.IntVar(value=40)
        tk.Spinbox(lrow, from_=0, to=45, width=3, textvariable=self.angle_deg_var).pack(side='left')
        ttk.Label(lrow, text='Tol').pack(side='left', padx=(6, 2))
        self.link_tol_var = tk.IntVar(value=10)
        tk.Spinbox(lrow, from_=1, to=300, width=4, textvariable=self.link_tol_var).pack(side='left')
        self._help(lrow, 'Connects pairs of fringe ends closer than Tol pixels whose joining line is '
                         'within Angle degrees of horizontal. 0 degrees means any angle.')
        mrow = ttk.Frame(ctrl)
        mrow.pack(anchor='w', fill='x', pady=1)
        self.btn_merge_overlay = ttk.Button(mrow, text='Merge Binaries', command=self.merge_overlay, state='disabled')
        self.btn_merge_overlay.pack(side='left')
        self._help(mrow, 'Copies the reference binary (at its current nudged position) into the shot '
                         'binary. Enabled when a reference binary is loaded.')

        # --- View ---
        self._section(ctrl, 'View', 'What is drawn on screen. None of these change your data or '
                                    'what gets saved.')
        self.show_shot_img = tk.BooleanVar(value=True)
        self.show_ref_img = tk.BooleanVar(value=True)
        self.show_shot_bin = tk.BooleanVar(value=True)
        self.show_ref_bin = tk.BooleanVar(value=True)
        self.shot_opacity_var = tk.DoubleVar(value=0.5)
        self.shot_brightness_var = tk.DoubleVar(value=0.0)
        self.ref_brightness_var = tk.DoubleVar(value=0.0)
        self.shot_binary_opacity_var = tk.DoubleVar(value=1.0)
        self.ref_binary_opacity_var = tk.DoubleVar(value=0.7)
        r = lambda: self._schedule_view()

        def layer_header(text, var, help_text):
            row = ttk.Frame(ctrl)
            row.pack(anchor='w', fill='x', pady=(4, 0))
            ttk.Checkbutton(row, text=text, variable=var, command=r).pack(side='left')
            self._help(row, help_text)

        layer_header('Shot image', self.show_shot_img, 'The shot interferogram. Opacity blends it over '
                                                       'the reference image. Brightness is relative to an '
                                                       'automatic contrast stretch done on load.')
        self._slider(ctrl, 'Opacity', self.shot_opacity_var, 0.0, 1.0, r)
        self._slider(ctrl, 'Brightness', self.shot_brightness_var, -2.0, 2.0, r, fmt='{:+.1f}')
        layer_header('Reference image', self.show_ref_img, 'The reference interferogram, drawn under the '
                                                           'shot image at its nudged position.')
        self._slider(ctrl, 'Brightness', self.ref_brightness_var, -2.0, 2.0, r, fmt='{:+.1f}')
        layer_header('Shot binary', self.show_shot_bin, 'The shot binary in black (gray mask in gray).')
        self._slider(ctrl, 'Opacity', self.shot_binary_opacity_var, 0.0, 1.0, r)
        layer_header('Reference binary', self.show_ref_bin, 'The reference binary in cyan.')
        self._slider(ctrl, 'Opacity', self.ref_binary_opacity_var, 0.0, 1.0, r)

        crow = ttk.Frame(ctrl)
        crow.pack(anchor='w', fill='x', pady=(6, 0))
        self.show_components_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(crow, text='Color comps', variable=self.show_components_var, command=r).pack(side='left')
        self._help(crow, 'Gives every connected fringe its own colour, so breaks and merges stand out.')
        mirow = ttk.Frame(ctrl)
        mirow.pack(anchor='w', fill='x', pady=(2, 0))
        self.mirror_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mirow, text='Mirror at x =', variable=self.mirror_var, command=r).pack(side='left')
        self.mirror_x_var = tk.StringVar(value='')
        tk.Entry(mirow, width=7, textvariable=self.mirror_x_var).pack(side='left', padx=(4, 0))
        self.mirror_x_var.trace_add('write', lambda *a: self._schedule_view())
        self._help(mirow, 'View only. Reflects the shot binary about this x (working-grid pixels) and '
                          'draws it in red under the black lines. Never saved.')
        rrow = ttk.Frame(ctrl)
        rrow.pack(anchor='w', fill='x', pady=(4, 0))
        ttk.Button(rrow, text='Reset Nudge', command=self.reset_nudge).pack(side='left')
        ttk.Button(rrow, text='Reset Crop', command=self.reset_crop).pack(side='left', padx=(6, 0))
        self._help(rrow, 'Reset Nudge puts the reference image and binary back where they loaded. '
                         'Reset Crop restores everything a crop cleared, keeping your other edits.')

        # --- footer ---
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=8)
        urow = ttk.Frame(ctrl)
        urow.pack(anchor='w', fill='x')
        ttk.Button(urow, text='Undo', command=self.undo).pack(side='left')
        ttk.Button(urow, text='Redo', command=self.redo).pack(side='left', padx=(6, 0))
        self._help(urow, 'Undo (Ctrl+Z) and Redo (Ctrl+Y or Ctrl+Shift+Z). Keeps the last 100 edits.')
        self.cursor_pos_var = tk.StringVar(value='X: ---, Y: ---')
        ttk.Label(ctrl, textvariable=self.cursor_pos_var, font=('Consolas', 9)).pack(anchor='w', pady=(6, 0))
        self.ruler_dist_var = tk.StringVar(value='Dist: 0.00 px')
        ttk.Label(ctrl, textvariable=self.ruler_dist_var, font=('Consolas', 9)).pack(anchor='w')
        ttk.Frame(ctrl, height=60).pack(fill='x')

        # --- right: status + canvas ---
        right = ttk.Frame(self)
        right.pack(side='left', fill='both', expand=True, padx=(0, 8), pady=8)
        self.status = ttk.Label(right, text='Open a binary (0/255) to edit, or use Send to Editor on the '
                                            'Detection tab', anchor='w')
        self.status.pack(side='bottom', fill='x')
        self.canvas = tk.Canvas(right, bg='gray20', highlightthickness=0)
        self.canvas.pack(side='top', fill='both', expand=True)

    def _bind_events(self):
        c = self.canvas
        c.bind('<Configure>', self._on_canvas_configure)
        c.bind('<Button-1>', self._on_left_down)
        c.bind('<B1-Motion>', self._on_left_drag)
        c.bind('<ButtonRelease-1>', self._on_left_up)
        c.bind('<Double-Button-1>', self._on_double_click)
        c.bind('<Button-3>', self._on_pan_start)
        c.bind('<B3-Motion>', self._on_pan_move)
        c.bind('<ButtonRelease-3>', self._on_pan_end)
        c.bind('<Motion>', self._on_mouse_move)
        c.bind('<Leave>', self._on_mouse_leave)
        c.bind('<MouseWheel>', self._on_mouse_wheel)
        c.bind('<Control-MouseWheel>', self._on_ctrl_mouse_wheel)
        c.bind('<Button-4>', self._on_mouse_wheel)
        c.bind('<Button-5>', self._on_mouse_wheel)
        c.bind('<Control-Button-4>', self._on_ctrl_mouse_wheel)
        c.bind('<Control-Button-5>', self._on_ctrl_mouse_wheel)
        top = self.winfo_toplevel()
        top.bind('<Control-z>', self._key(lambda e: self.undo()), add='+')
        top.bind('<Control-y>', self._key(lambda e: self.redo()), add='+')
        top.bind('<Control-Z>', self._key(lambda e: self.redo()), add='+')
        top.bind('<Control-Shift-Z>', self._key(lambda e: self.redo()), add='+')
        for key, label, sc, tip in TOOLS:
            for k in (sc.lower(), sc.upper()):
                top.bind(f'<Key-{k}>', self._key(lambda e, t=key: self._set_tool(t), plain=True), add='+')
        top.bind('<Left>', self._key(lambda e: self._nudge_overlay(-1, 0)), add='+')
        top.bind('<Right>', self._key(lambda e: self._nudge_overlay(1, 0)), add='+')
        top.bind('<Up>', self._key(lambda e: self._nudge_overlay(0, -1)), add='+')
        top.bind('<Down>', self._key(lambda e: self._nudge_overlay(0, 1)), add='+')
        top.bind('<Escape>', self._key(lambda e: self._cancel_shape()), add='+')
        top.bind('<Return>', self._key(lambda e: self._finish_spline()), add='+')

    def _key(self, fn, plain=False):
        """Only act when the Editor is on screen and the user is not typing in a box."""
        def handler(event):
            try:
                if not self.winfo_ismapped():
                    return
                w = self.focus_get()
                cls = w.winfo_class() if w is not None else ''
                if cls in ('Entry', 'TEntry', 'Spinbox', 'TSpinbox', 'TCombobox', 'Text'):
                    return
                if plain and (event.state & 0x4):      # ignore Ctrl+letter for tool keys
                    return
            except Exception:
                pass
            fn(event)
            return 'break'
        return handler

    # ============================================================ helpers ==
    def set_status(self, text):
        try:
            self.status.config(text=text)
        except Exception:
            pass

    def _initial_dir(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        edited_dir = os.path.join(base_dir, 'EditedImages')
        return edited_dir if os.path.isdir(edited_dir) else base_dir

    def _layer(self, name):
        return {'mask': self.mask, 'gray': self.gray_mask, 'ovl': self.overlay_mask,
                'shot': self._shot_img, 'ref': self._ref_img}[name]

    def _set_layer(self, name, arr):
        if name == 'mask':
            self.mask = arr
        elif name == 'gray':
            self.gray_mask = arr
        elif name == 'ovl':
            self.overlay_mask = arr
        elif name == 'shot':
            self._shot_img = arr
        elif name == 'ref':
            self._ref_img = arr

    def _edit_target(self):
        """(layer name, x offset, y offset) of the binary the tools edit."""
        if self.edit_layer_var.get() == 'ref':
            if self.overlay_mask is None:
                return None
            return 'ovl', self.overlay_pos[0], self.overlay_pos[1]
        return 'mask', 0, 0

    def _invalidate(self, name=None):
        """Mark a layer as changed: bumps its version (drops cached zoom levels),
        clears cached fringe ends, and for the shot binary the component labels."""
        names = [name] if name else ['mask', 'gray', 'ovl', 'shot', 'ref']
        if not name:
            self._pyr.clear()
        for n in names:
            self._ver[n] = self._ver.get(n, 0) + 1
            self._ends_cache.pop(n, None)
        if 'mask' in names:
            self._labels_dirty = True
            self.magic2_labels = set()

    # =================================================== undo / redo ======
    def _begin(self, names):
        names = [n for n in names if self._layer(n) is not None]
        self._snap_before = {n: self._layer(n).copy() for n in names}

    def _commit(self, label='edit'):
        snap = self._snap_before
        self._snap_before = None
        if not snap:
            return False
        entry = []
        for name, before in snap.items():
            after = self._layer(name)
            if after is None or after.shape != before.shape:
                continue
            diff = before != after
            rows = np.flatnonzero(diff.any(axis=1))
            if rows.size == 0:
                continue
            cols = np.flatnonzero(diff[rows[0]:rows[-1] + 1].any(axis=0))
            y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            entry.append((name, y0, x0, before[y0:y1, x0:x1].copy(), after[y0:y1, x0:x1].copy()))
            self._invalidate(name)
            if name in ('mask', 'gray'):
                self._dirty = True
        if not entry:
            return False
        self._undo.append((label, entry))
        if len(self._undo) > self.MAX_UNDO:
            self._undo = self._undo[-self.MAX_UNDO:]
        self._redo.clear()
        return True

    def _apply_entry(self, entry, use_before):
        for name, y0, x0, before, after in entry:
            arr = self._layer(name)
            if arr is None:
                continue
            patch = before if use_before else after
            h, w = patch.shape[:2]
            if y0 + h > arr.shape[0] or x0 + w > arr.shape[1]:
                continue
            arr[y0:y0 + h, x0:x0 + w] = patch
            self._invalidate(name)
            if name in ('mask', 'gray'):
                self._dirty = True

    def undo(self):
        if not self._undo:
            self.set_status('Nothing to undo')
            return
        label, entry = self._undo.pop()
        self._apply_entry(entry, use_before=True)
        self._redo.append((label, entry))
        self.set_status(f'Undid: {label}')
        self._schedule_view()

    def redo(self):
        if not self._redo:
            self.set_status('Nothing to redo')
            return
        label, entry = self._redo.pop()
        self._apply_entry(entry, use_before=False)
        self._undo.append((label, entry))
        self.set_status(f'Redid: {label}')
        self._schedule_view()

    def _clear_history(self):
        self._undo.clear()
        self._redo.clear()
        self._snap_before = None

    # ========================================================= grid fit ==
    def _fit_image_to_grid(self, img):
        """Continuous-tone plate onto the working grid (cubic up, area down)."""
        if img is None:
            return None
        ih, iw = img.shape[:2]
        if (ih, iw) == (self.grid, self.grid):
            return img
        up = self.grid > max(ih, iw)
        interp = cv2.INTER_CUBIC if up else cv2.INTER_AREA
        if ih == iw:
            return cv2.resize(img, (self.grid, self.grid), interpolation=interp)
        f = min(self.grid / float(ih), self.grid / float(iw))
        nh, nw = max(1, int(round(ih * f))), max(1, int(round(iw * f)))
        scaled = cv2.resize(img, (nw, nh), interpolation=interp)
        out = np.zeros((self.grid, self.grid) + img.shape[2:], dtype=img.dtype)
        out[:nh, :nw] = scaled
        return out

    def _fit_binary_to_grid(self, layer, fill):
        """Exact integer-factor smaller binaries are nearest-upsampled (pure pixel
        replication, cannot fuse fringes); anything else is pasted top-left."""
        out = np.full((self.grid, self.grid), fill, dtype=np.uint8)
        if layer is None:
            return out
        ih, iw = layer.shape[:2]
        if ih == self.grid and iw == self.grid:
            return layer.astype(np.uint8).copy()
        if ih == iw and ih > 0 and self.grid % ih == 0:
            return cv2.resize(layer.astype(np.uint8), (self.grid, self.grid), interpolation=cv2.INTER_NEAREST)
        ph, pw = min(self.grid, ih), min(self.grid, iw)
        out[:ph, :pw] = layer[:ph, :pw]
        return out

    def set_grid_factor(self, factor, rescale_existing=True):
        """Switch the working grid between 2048 (factor=1) and 4096 (factor=2).
        Going up is exact pixel replication. Going down clears the masks after
        asking, because shrinking a skeleton fuses fringes."""
        factor = max(1, int(factor))
        if factor == self.grid_factor:
            return True
        old_grid = self.grid
        new_grid = 2048 * factor
        if new_grid < old_grid and rescale_existing:
            if not messagebox.askokcancel(
                    'Reduce grid?',
                    'Going from {0}x{0} back to {1}x{1} cannot preserve the mask: shrinking a skeleton '
                    'fuses fringes that are currently separate.\n\nThe mask layers will be CLEARED. Save '
                    'your work first if you need it.'.format(old_grid, new_grid)):
                return False
            rescale_existing = False

        def _resize(a, binary=True):
            if a is None:
                return None
            return cv2.resize(a, (new_grid, new_grid),
                              interpolation=cv2.INTER_NEAREST if binary else cv2.INTER_CUBIC)

        self.grid_factor = factor
        self.grid = new_grid
        sf = new_grid / float(old_grid)
        if rescale_existing:
            self.mask = _resize(self.mask)
            self.gray_mask = _resize(self.gray_mask)
            self._crop_mask = _resize(self._crop_mask)
            if self.overlay_mask is not None:
                oh, ow = self.overlay_mask.shape[:2]
                self.overlay_mask = cv2.resize(self.overlay_mask, (int(ow * sf), int(oh * sf)),
                                               interpolation=cv2.INTER_NEAREST)
            self._shot_img = _resize(self._shot_img, binary=False)
            self._ref_img = _resize(self._ref_img, binary=False)
            self._uncropped = {k: (_resize(v, binary=k in ('mask', 'gray', 'ovl')) if k != 'ovl' or v is None
                                   else cv2.resize(v, (int(v.shape[1] * sf), int(v.shape[0] * sf)),
                                                   interpolation=cv2.INTER_NEAREST))
                               for k, v in self._uncropped.items()}
            self.ref_pos = (int(self.ref_pos[0] * sf), int(self.ref_pos[1] * sf))
            self.overlay_pos = (int(self.overlay_pos[0] * sf), int(self.overlay_pos[1] * sf))
            self._initial_ref_pos = (int(self._initial_ref_pos[0] * sf), int(self._initial_ref_pos[1] * sf))
            self._initial_overlay_pos = (int(self._initial_overlay_pos[0] * sf), int(self._initial_overlay_pos[1] * sf))
            self._zoom /= sf
            self._ox *= sf
            self._oy *= sf
        else:
            self.mask = np.full((self.grid, self.grid), 255, dtype=np.uint8)
            self.gray_mask = np.zeros((self.grid, self.grid), dtype=np.uint8)
            self._crop_mask = np.full((self.grid, self.grid), 255, dtype=np.uint8)
            self.overlay_mask = None
            self.overlay_active = False
            self._uncropped = {}
            self._shot_img = None
            self._ref_img = None
            self.ref_pos = (0, 0)
            self.overlay_pos = (0, 0)
            self._fitted = False
            self._dirty = False
            try:
                self.btn_merge_overlay.config(state='disabled')
            except Exception:
                pass
        self._clear_history()
        self._invalidate()
        self._ruler_points = []
        if not self._fitted:
            self.fit_view()
        self._schedule_view()
        self.set_status(f'Working grid: {self.grid}x{self.grid}')
        return True

    # ======================================================= data in/out ==
    def set_data(self, mask, background=None):
        if mask is None:
            return
        m = np.asarray(mask)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        self.mask = self._fit_binary_to_grid(np.where(m < 128, 0, 255).astype(np.uint8), 255)
        self.gray_mask = self._fit_binary_to_grid(np.where((m > 0) & (m < 255), 255, 0).astype(np.uint8), 0)
        if background is not None:
            self._shot_img = self._fit_image_to_grid(self._to_display(background))
        self._after_new_shot()

    def receive_from_detection(self, binary, shot_img, saved_path=None, name=''):
        """Called by the Detection tab's Send to Editor button."""
        if self._dirty and (self._undo or self._redo):
            if not messagebox.askyesno('Replace shot binary?',
                                       'The Editor has edits to the current shot binary that have not '
                                       'been saved. Replace it with the new detection?'):
                return False
        self.set_data(binary, shot_img)
        self._dirty = False
        self.set_status('Received {} from Detection{}'.format(
            name or 'binary', f' (saved as {os.path.basename(saved_path)})' if saved_path else ''))
        return True

    def get_mask(self):
        if self.mask is None:
            return None
        result = self.mask.copy()
        if self.gray_mask is not None:
            result = np.where(self.gray_mask > 0, 128, result)
        return result

    def _after_new_shot(self):
        self._crop_mask = np.full((self.grid, self.grid), 255, dtype=np.uint8)
        self._uncropped = {}
        self._clear_history()
        self._invalidate()
        if not self._fitted:
            self.after(30, self.fit_view)
        self._schedule_view()

    @staticmethod
    def _read_any(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _to_display(img):
        """Photo plate -> uint8 with an automatic contrast stretch (0.5 / 99.5 percentile),
        so a few hot pixels cannot make the whole image dark."""
        a = np.asarray(img).astype(np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max())
        if hi <= lo:
            return np.zeros(a.shape, np.uint8)
        return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    @staticmethod
    def _to_binary8(img):
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    def _ask_open(self, title):
        return filedialog.askopenfilename(parent=self.winfo_toplevel(), title=title, initialdir=self._initial_dir(),
                                          filetypes=[('Images', ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'))])

    def open_binary(self):
        path = self._ask_open('Open shot binary')
        if not path:
            return
        img = self._read_any(path)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image')
            return
        img = self._to_binary8(img)
        self.mask = self._fit_binary_to_grid(np.where(img < 128, 0, 255).astype(np.uint8), 255)
        self.gray_mask = self._fit_binary_to_grid(np.where((img > 0) & (img < 255), 255, 0).astype(np.uint8), 0)
        self._dirty = False
        self._after_new_shot()
        gray_px = int(np.sum(self.gray_mask > 0))
        msg = f'Loaded: {os.path.basename(path)} - {self.mask.shape[1]}x{self.mask.shape[0]}'
        if gray_px:
            msg += f' (with {gray_px} gray mask pixels)'
        self.set_status(msg)

    def open_overlay_binary(self):
        path = self._ask_open('Open reference binary')
        if not path:
            return
        img = self._read_any(path)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image')
            return
        img = self._to_binary8(img)
        ov = np.where(img < 128, 0, 255).astype(np.uint8)
        oh, ow = ov.shape[:2]
        if oh == ow and oh < self.grid and self.grid % oh == 0:
            ov = cv2.resize(ov, (self.grid, self.grid), interpolation=cv2.INTER_NEAREST)
        self.overlay_mask = ov
        oh, ow = ov.shape[:2]
        self.overlay_pos = ((self.grid - ow) // 2, (self.grid - oh) // 2)
        self._initial_overlay_pos = self.overlay_pos
        self.overlay_active = True
        self._uncropped.pop('ovl', None)
        self.btn_merge_overlay.config(state='normal')
        self._invalidate('ovl')
        self.set_status('Reference binary loaded. Nudge with the arrow keys or Ctrl + right-drag, '
                        'then Merge Binaries if needed.')
        self._schedule_view()

    def _load_plate(self, which):
        path = filedialog.askopenfilename(parent=self.winfo_toplevel(),
                                          title='Select Shot Image' if which == 'shot' else 'Select Reference Image',
                                          filetypes=[('Images', ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'))])
        if not path:
            return
        img = self._read_any(path)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image')
            return
        nh, nw = img.shape[:2]
        disp = self._fit_image_to_grid(self._to_display(img))
        if which == 'shot':
            self._shot_img = disp
            self._uncropped.pop('shot', None)
            self._invalidate('shot')
        else:
            self._ref_img = disp
            self.ref_pos = (0, 0)
            self._initial_ref_pos = (0, 0)
            self._uncropped.pop('ref', None)
            self._invalidate('ref')
        note = f' ({nw}x{nh} -> {disp.shape[1]}x{disp.shape[0]})' if (nh, nw) != disp.shape[:2] else ''
        self.set_status(f'Loaded {"Shot" if which == "shot" else "Reference"}: {os.path.basename(path)}{note}')
        if not self._fitted:
            self.after(30, self.fit_view)
        self._schedule_view()

    def _load_shot(self):
        self._load_plate('shot')

    def _load_reference(self):
        self._load_plate('ref')

    def open_gray_mask(self):
        path = self._ask_open('Open gray mask')
        if not path:
            return
        img = self._read_any(path)
        if img is None:
            messagebox.showerror('Open error', 'Failed to read the image')
            return
        img = self._to_binary8(img)
        if img.shape != (self.grid, self.grid):
            img = cv2.resize(img, (self.grid, self.grid), interpolation=cv2.INTER_NEAREST)
        self._begin(['gray'])
        self.gray_mask = np.where(img > 127, 255, 0).astype(np.uint8)
        # whole-layer replacement: keep it undoable
        self._snap_before = {'gray': self._snap_before['gray']} if self._snap_before else None
        self._commit('Open gray mask')
        self.set_status(f'Loaded gray mask: {os.path.basename(path)}')
        self._schedule_view()

    def _ask_save(self, title):
        d = self._initial_dir()
        try:
            os.makedirs(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'EditedImages'), exist_ok=True)
            d = self._initial_dir()
        except Exception:
            pass
        return filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title=title, initialdir=d,
                                            defaultextension='.png',
                                            filetypes=[('PNG', '*.png'), ('TIFF', '*.tif'), ('JPEG', '*.jpg')])

    def save_binary(self):
        if self.mask is None:
            messagebox.showinfo('Nothing to save', 'Load an image first')
            return
        path = self._ask_save('Save shot binary')
        if not path:
            return
        try:
            if not cv2.imwrite(path, self.get_mask()):
                raise RuntimeError('cv2.imwrite returned False')
            self._dirty = False
            self.set_status(f'Saved shot binary: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def save_reference_binary(self):
        if self.overlay_mask is None:
            messagebox.showinfo('Nothing to save', 'No reference binary loaded')
            return
        path = self._ask_save('Save reference binary')
        if not path:
            return
        try:
            if not cv2.imwrite(path, self.overlay_mask):
                raise RuntimeError('cv2.imwrite returned False')
            self.set_status(f'Saved reference binary: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def save_gray_mask(self):
        if self.gray_mask is None:
            messagebox.showinfo('Nothing to save', 'No gray mask present')
            return
        path = self._ask_save('Save gray mask')
        if not path:
            return
        try:
            if not cv2.imwrite(path, self.gray_mask):
                raise RuntimeError('cv2.imwrite returned False')
            self.set_status(f'Saved gray mask: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    # ============================================================ tools ==
    def _set_tool(self, mode):
        if mode not in dict((t[0], t) for t in TOOLS):
            return
        if self._painting:
            return
        self._clear_preview()
        self._tool_points = []
        if mode != 'ruler':
            self._ruler_points = []
            self._redraw_ruler()
        if mode != 'check' and self.magic2_labels:
            self.magic2_labels = set()
            self._schedule_view()
        old = self.mode_var.get()
        if old in SIZED_TOOLS:
            try:
                self._sizes[old] = float(self.brush_var.get())
            except Exception:
                pass
        self.mode_var.set(mode)
        if mode in SIZED_TOOLS:
            self.brush_var.set(self._sizes.get(mode, DEFAULT_SIZES[mode]))
        # contextual options
        for w in (self._opt_size, self._opt_snap, self._opt_ruler, self._opt_hint):
            w.pack_forget()
        hints = {
            'pen': None, 'erase': None,
            'mask': 'Hold Shift while painting to remove mask.',
            'line': None, 'spline': 'Finish with Ctrl+click, double click or Enter. Esc cancels.',
            'delete': 'Click a line to delete the whole connected piece.',
            'crop': 'Drag a box. Everything outside it is cleared.',
            'ruler': None, 'check': 'Drag a line across the fringes to test.',
        }
        if mode in SIZED_TOOLS:
            self._opt_size.pack(anchor='w', fill='x')
        if mode in ('line', 'spline'):
            self._opt_snap.pack(anchor='w', fill='x')
        if mode == 'ruler':
            self._opt_ruler.pack(anchor='w', fill='x')
        if hints.get(mode):
            self._opt_hint.config(text=hints[mode])
            self._opt_hint.pack(anchor='w', fill='x', pady=(2, 0))
        label = [t[1] for t in TOOLS if t[0] == mode][0]
        self.set_status(f'Tool: {label}')
        self._update_cursor()
        self._update_snap_marker()

    def _on_edit_layer(self):
        if self.edit_layer_var.get() == 'ref' and self.overlay_mask is None:
            self.set_status('No reference binary loaded: Open > Reference Binary first')
        else:
            self.set_status('Now editing the ' + ('reference binary (cyan)' if self.edit_layer_var.get() == 'ref'
                                                  else 'shot binary (black)'))
        self._update_cursor()
        self._update_snap_marker()

    def _on_size_change(self):
        if self.mode_var.get() in SIZED_TOOLS:
            self._sizes[self.mode_var.get()] = float(self.brush_var.get())
        self._update_cursor()

    # ----------------------------------------------------- coordinates ---
    def _to_img(self, cx, cy):
        return self._ox + cx / self._zoom, self._oy + cy / self._zoom

    def _to_canvas(self, ix, iy):
        return (ix - self._ox) * self._zoom, (iy - self._oy) * self._zoom

    # ------------------------------------------------------- painting ---
    @staticmethod
    def _paint_disc(arr, xi, yi, r, value):
        """Set pixels whose centre lies within r of (xi, yi). Always hits the pixel under (xi, yi)."""
        h, w = arr.shape[:2]
        x0 = int(max(0, math.floor(xi - r))); x1 = int(min(w - 1, math.ceil(xi + r)))
        y0 = int(max(0, math.floor(yi - r))); y1 = int(min(h - 1, math.ceil(yi + r)))
        if x1 < x0 or y1 < y0:
            return
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        disc = (xx + 0.5 - xi) ** 2 + (yy + 0.5 - yi) ** 2 <= r * r
        px, py = int(math.floor(xi)), int(math.floor(yi))
        if x0 <= px <= x1 and y0 <= py <= y1:
            disc[py - y0, px - x0] = True
        arr[y0:y1 + 1, x0:x1 + 1][disc] = value

    def _paint_segment(self, layer, ox, oy, a, b, r, value):
        arr = self._layer(layer)
        if arr is None:
            return
        (x0, y0), (x1, y1) = a, b
        dist = math.hypot(x1 - x0, y1 - y0)
        step = max(0.25, r * 0.3)
        n = max(1, int(math.ceil(dist / step)))
        for i in range(n + 1):
            t = i / float(n)
            self._paint_disc(arr, x0 + (x1 - x0) * t - ox, y0 + (y1 - y0) * t - oy, r, value)

    def _stroke_target(self, event):
        mode = self.mode_var.get()
        if mode == 'mask':
            return 'gray', 0, 0, (0 if (event.state & 0x1) else 255)
        tgt = self._edit_target()
        if tgt is None:
            return None
        name, ox, oy = tgt
        return name, ox, oy, (0 if mode == 'pen' else 255)

    def _redraw_canvas_rect(self, ia, ib, r):
        """Recomposite only the screen rectangle around an image-space segment."""
        (xa, ya), (xb, yb) = ia, ib
        ca = self._to_canvas(min(xa, xb) - r - 2, min(ya, yb) - r - 2)
        cb = self._to_canvas(max(xa, xb) + r + 2, max(ya, yb) + r + 2)
        self._render_rect(int(math.floor(ca[0])), int(math.floor(ca[1])), int(math.ceil(cb[0])) + 1, int(math.ceil(cb[1])) + 1)

    # ============================================================ mouse ==
    def _on_left_down(self, event):
        mode = self.mode_var.get()
        ix, iy = self._to_img(event.x, event.y)
        if mode in SIZED_TOOLS:
            tgt = self._stroke_target(event)
            if tgt is None:
                self.set_status('No reference binary loaded: Open > Reference Binary first')
                return
            name, ox, oy, value = tgt
            self._begin([name])
            self._paint_layer = (name, ox, oy, value)
            r = max(0.5, float(self.brush_var.get()))
            self._paint_disc(self._layer(name), ix - ox, iy - oy, r, value)
            self._pyr_patch(name, ix - ox - r - 1, iy - oy - r - 1, ix - ox + r + 2, iy - oy + r + 2)
            self._painting = True
            self._last_paint_xy = (ix, iy)
            self._redraw_canvas_rect((ix, iy), (ix, iy), r)
        elif mode == 'delete':
            self._delete_fringe_at(ix, iy)
        elif mode == 'ruler':
            self._ruler_down(ix, iy)
        elif mode in ('line', 'check', 'crop'):
            p = self._snap((ix, iy)) if mode == 'line' else (ix, iy)
            if not self._tool_points:
                self._tool_points = [p]
                self._drag_start = (event.x, event.y)
            else:
                self._tool_points.append(p)
                self._commit_shape()
        elif mode == 'spline':
            if (event.state & 0x4) and len(self._tool_points) >= 2:
                self._finish_spline()
            else:
                self._tool_points.append(self._snap((ix, iy)))
                self._update_preview(event.x, event.y)
        self._last_mouse = (event.x, event.y)

    def _on_left_drag(self, event):
        mode = self.mode_var.get()
        self._last_mouse = (event.x, event.y)
        ix, iy = self._to_img(event.x, event.y)
        self._update_coords(ix, iy)
        if mode in SIZED_TOOLS and self._painting:
            name, ox, oy, value = self._paint_layer
            r = max(0.5, float(self.brush_var.get()))
            self._paint_segment(name, ox, oy, self._last_paint_xy, (ix, iy), r, value)
            (ax, ay), (bx, by) = self._last_paint_xy, (ix, iy)
            self._pyr_patch(name, min(ax, bx) - ox - r - 1, min(ay, by) - oy - r - 1,
                            max(ax, bx) - ox + r + 2, max(ay, by) - oy + r + 2)
            self._redraw_canvas_rect(self._last_paint_xy, (ix, iy), r)
            self._last_paint_xy = (ix, iy)
            self._update_cursor()
        elif mode == 'ruler':
            self._ruler_drag(ix, iy)
        elif mode in ('line', 'check', 'crop', 'spline') and self._tool_points:
            self._update_preview(event.x, event.y)

    def _on_left_up(self, event):
        mode = self.mode_var.get()
        if mode in SIZED_TOOLS and self._painting:
            self._painting = False
            self._last_paint_xy = None
            name = self._paint_layer[0]
            self._paint_layer = None
            label = {'pen': 'Pen stroke', 'erase': 'Eraser stroke', 'mask': 'Mask stroke'}[mode]
            self._commit(label)
            if self.show_components_var.get() and name == 'mask':
                self._schedule_view()
            return
        if mode == 'ruler':
            self._ruler_up(*self._to_img(event.x, event.y))
            return
        if mode in ('line', 'check', 'crop') and len(self._tool_points) == 1 and self._drag_start is not None:
            dx = event.x - self._drag_start[0]; dy = event.y - self._drag_start[1]
            if math.hypot(dx, dy) > 3:                 # click-and-drag finishes the shape
                ix, iy = self._to_img(event.x, event.y)
                self._tool_points.append(self._snap((ix, iy)) if mode == 'line' else (ix, iy))
                self._commit_shape()

    def _on_double_click(self, event):
        if self.mode_var.get() == 'spline' and len(self._tool_points) >= 2:
            self._finish_spline()

    def _on_pan_start(self, event):
        if event.state & 0x4 and (self.overlay_mask is not None or self._ref_img is not None):
            self._moving_overlay = True
            self._pan_start = (event.x, event.y, self.overlay_pos, self.ref_pos)
            return 'break'
        self._pan_start = (event.x, event.y, self._ox, self._oy)
        try:
            self.canvas.configure(cursor='fleur')
        except Exception:
            pass
        return 'break'

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        if self._moving_overlay:
            x0, y0, opos, rpos = self._pan_start
            dx = int(round((event.x - x0) / self._zoom)); dy = int(round((event.y - y0) / self._zoom))
            if self.overlay_mask is not None:
                self.overlay_pos = (opos[0] + dx, opos[1] + dy)
            if self._ref_img is not None:
                self.ref_pos = (rpos[0] + dx, rpos[1] + dy)
            self._invalidate('ovl')
            self._schedule_view()
            return 'break'
        x0, y0, ox, oy = self._pan_start
        self._ox = ox - (event.x - x0) / self._zoom
        self._oy = oy - (event.y - y0) / self._zoom
        self._schedule_view(fast=True)
        return 'break'

    def _on_pan_end(self, event):
        self._pan_start = None
        self._moving_overlay = False
        try:
            self.canvas.configure(cursor='')
        except Exception:
            pass
        return 'break'

    def _on_mouse_move(self, event):
        self._last_mouse = (event.x, event.y)
        ix, iy = self._to_img(event.x, event.y)
        self._update_coords(ix, iy)
        self._update_cursor()
        mode = self.mode_var.get()
        if mode in ('line', 'spline', 'check', 'crop') and self._tool_points:
            self._update_preview(event.x, event.y)
        if mode == 'ruler' and len(self._ruler_points) == 1:
            self._redraw_ruler()
        self._update_snap_marker()

    def _on_mouse_leave(self, event):
        self._last_mouse = None
        for item in (self._cursor_item, self._snap_item):
            if item is not None:
                self.canvas.delete(item)
        self._cursor_item = None
        self._snap_item = None
        self.cursor_pos_var.set('X: ---, Y: ---')

    def _update_coords(self, ix, iy):
        self.cursor_pos_var.set(f'X: {int(math.floor(ix))}, Y: {int(math.floor(iy))}')

    def _on_mouse_wheel(self, event):
        if getattr(event, 'state', 0) & 0x4:
            return
        delta = int(getattr(event, 'delta', 0))
        if delta == 0:
            num = getattr(event, 'num', None)
            delta = 120 if num == 4 else (-120 if num == 5 else 0)
        if delta == 0:
            return
        f = 1.15 if delta > 0 else 1 / 1.15
        nz = max(0.02, min(64.0, self._zoom * f))
        ix, iy = self._to_img(event.x, event.y)
        self._zoom = nz
        self._ox = ix - event.x / nz
        self._oy = iy - event.y / nz
        self._schedule_view(fast=True)

    def _on_ctrl_mouse_wheel(self, event):
        delta = int(getattr(event, 'delta', 0))
        if delta == 0:
            num = getattr(event, 'num', None)
            delta = 120 if num == 4 else -120
        if self.mode_var.get() not in SIZED_TOOLS:
            return 'break'
        f = 1.1 if delta > 0 else 1 / 1.1
        self.brush_var.set(round(max(0.5, min(60.0, float(self.brush_var.get()) * f)), 2))
        self._on_size_change()
        return 'break'

    def _on_canvas_configure(self, event=None):
        if not self._fitted:
            self.fit_view()
        self._schedule_view()

    def fit_view(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        z = min(cw / float(self.grid), ch / float(self.grid))
        self._zoom = max(0.02, z)
        self._ox = (self.grid - cw / self._zoom) / 2.0
        self._oy = (self.grid - ch / self._zoom) / 2.0
        self._fitted = True
        self._schedule_view()

    # ---------------------------------------------------- cursor / snap ---
    def _update_cursor(self):
        mode = self.mode_var.get()
        if self._last_mouse is None or mode not in SIZED_TOOLS:
            if self._cursor_item is not None:
                self.canvas.delete(self._cursor_item)
                self._cursor_item = None
            return
        x, y = self._last_mouse
        r = max(1.0, float(self.brush_var.get()) * self._zoom)
        if mode == 'mask':
            color = '#888888'
        elif mode == 'pen':
            color = '#00aaff' if self.edit_layer_var.get() == 'ref' else '#00ff88'
        else:
            color = '#ff5555'
        if self._cursor_item is None:
            self._cursor_item = self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, width=1)
        else:
            self.canvas.coords(self._cursor_item, x - r, y - r, x + r, y + r)
            self.canvas.itemconfig(self._cursor_item, outline=color)
        self.canvas.tag_raise(self._cursor_item)

    def _fringe_ends(self, name):
        if name in self._ends_cache:
            return self._ends_cache[name]
        arr = self._layer(name)
        if arr is None:
            return np.zeros(0, int), np.zeros(0, int)
        ink = (arr == 0).astype(np.uint8)
        nb = cv2.filter2D(ink, cv2.CV_8U, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8),
                          borderType=cv2.BORDER_CONSTANT)
        ys, xs = np.nonzero(ink & (nb <= 1))       # line ends and isolated pixels
        self._ends_cache[name] = (ys, xs)
        return ys, xs

    def _snap_target(self, ix, iy):
        if not self.snap_var.get() or self.mode_var.get() not in ('line', 'spline'):
            return None
        tgt = self._edit_target()
        if tgt is None:
            return None
        name, ox, oy = tgt
        ys, xs = self._fringe_ends(name)
        if len(ys) == 0:
            return None
        rad = 12.0 / self._zoom
        lx, ly = ix - ox, iy - oy
        sel = (np.abs(xs + 0.5 - lx) <= rad) & (np.abs(ys + 0.5 - ly) <= rad)
        if not sel.any():
            return None
        cx, cy = xs[sel] + 0.5, ys[sel] + 0.5
        d = (cx - lx) ** 2 + (cy - ly) ** 2
        k = int(np.argmin(d))
        if d[k] > rad * rad:
            return None
        return float(cx[k] + ox), float(cy[k] + oy)

    def _snap(self, p):
        s = self._snap_target(*p)
        return s if s is not None else p

    def _update_snap_marker(self):
        s = None
        if self._last_mouse is not None:
            s = self._snap_target(*self._to_img(*self._last_mouse))
        if s is None:
            if self._snap_item is not None:
                self.canvas.delete(self._snap_item)
                self._snap_item = None
            return
        cx, cy = self._to_canvas(*s)
        r = 6
        if self._snap_item is None:
            self._snap_item = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline='#00ff00', width=2)
        else:
            self.canvas.coords(self._snap_item, cx - r, cy - r, cx + r, cy + r)
        self.canvas.tag_raise(self._snap_item)

    # ------------------------------------------------------- shapes -----
    def _clear_preview(self):
        for item in self._preview_items:
            try:
                self.canvas.delete(item)
            except Exception:
                pass
        self._preview_items = []

    def _cancel_shape(self):
        if self._tool_points:
            self._tool_points = []
            self._clear_preview()
            self.set_status('Cancelled')

    def _update_preview(self, mx, my):
        self._clear_preview()
        if not self._tool_points:
            return
        mode = self.mode_var.get()
        cur = self._to_img(mx, my)
        if mode in ('line', 'spline'):
            cur = self._snap(cur)
        if mode == 'crop':
            a = self._to_canvas(*self._tool_points[0]); b = self._to_canvas(*cur)
            self._preview_items.append(self.canvas.create_rectangle(a[0], a[1], b[0], b[1], outline='#ff00ff',
                                                                    dash=(4, 4), width=2))
            return
        pts = self._tool_points + [cur]
        if mode == 'spline':
            pts = self._calculate_spline(pts)
        coords = []
        for px, py in pts:
            coords.extend(self._to_canvas(px, py))
        color = '#ff8800' if mode == 'check' else '#00ff88'
        if len(coords) >= 4:
            self._preview_items.append(self.canvas.create_line(*coords, fill=color, width=1, dash=(4, 4)))
        for px, py in self._tool_points:
            cx, cy = self._to_canvas(px, py)
            self._preview_items.append(self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill='#ff00ff', outline=''))

    def _finish_spline(self):
        if self.mode_var.get() == 'spline' and len(self._tool_points) >= 2:
            self._commit_shape()

    @staticmethod
    def _calculate_spline(points):
        if len(points) < 2:
            return points
        pts = np.array(points, dtype=np.float64)
        dist = np.sum(np.diff(pts, axis=0) ** 2, axis=1)
        pts = pts[np.insert(dist > 1e-4, 0, True)]
        if len(pts) == 2:
            return pts
        k = 3 if len(pts) > 3 else 2
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=k)
            u_new = np.linspace(0, 1, max(100, len(pts) * 20))
            x_new, y_new = splev(u_new, tck)
            return np.column_stack((x_new, y_new))
        except Exception:
            return pts

    def _commit_shape(self):
        mode = self.mode_var.get()
        pts = self._tool_points
        self._tool_points = []
        self._drag_start = None
        self._clear_preview()
        if len(pts) < 2:
            return
        if mode == 'crop':
            self._apply_crop(pts[0], pts[1])
            return
        if mode == 'check':
            self._connection_check(pts[0], pts[1])
            return
        tgt = self._edit_target()
        if tgt is None:
            self.set_status('No reference binary loaded: Open > Reference Binary first')
            return
        name, ox, oy = tgt
        arr = self._layer(name)
        self._begin([name])
        if mode == 'line':
            p1 = (int(math.floor(pts[0][0] - ox)), int(math.floor(pts[0][1] - oy)))
            p2 = (int(math.floor(pts[1][0] - ox)), int(math.floor(pts[1][1] - oy)))
            if p1 == p2:
                self._snap_before = None
                self.set_status('Points were identical. Cancelled.')
                return
            cv2.line(arr, p1, p2, 0, 1)
        else:
            curve = np.array(self._calculate_spline(pts), dtype=np.float64)
            curve[:, 0] -= ox
            curve[:, 1] -= oy
            curve = np.floor(curve).astype(np.int32)
            cv2.polylines(arr, [curve], isClosed=False, color=0, thickness=1)
        self._commit('Line' if mode == 'line' else 'Spline')
        self.set_status(('Line' if mode == 'line' else 'Spline') + ' drawn.')
        self._schedule_view()

    # ------------------------------------------------- delete fringe ----
    def _delete_fringe_at(self, ix, iy):
        tgt = self._edit_target()
        if tgt is None:
            self.set_status('No reference binary loaded: Open > Reference Binary first')
            return
        name, ox, oy = tgt
        arr = self._layer(name)
        lx, ly = ix - ox, iy - oy
        h, w = arr.shape
        rad = max(2, int(math.ceil(6.0 / self._zoom)))
        x0, x1 = max(0, int(lx) - rad), min(w, int(lx) + rad + 1)
        y0, y1 = max(0, int(ly) - rad), min(h, int(ly) + rad + 1)
        ys, xs = np.nonzero(arr[y0:y1, x0:x1] == 0)
        if len(ys) == 0:
            self.set_status('No fringe under the cursor')
            return
        d = (xs + x0 + 0.5 - lx) ** 2 + (ys + y0 + 0.5 - ly) ** 2
        k = int(np.argmin(d))
        sx, sy = int(xs[k] + x0), int(ys[k] + y0)
        self._begin([name])
        ink = (arr == 0).astype(np.uint8)
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ink, ff, (sx, sy), 2, flags=8)
        comp = ink == 2
        n_px = int(comp.sum())
        arr[comp] = 255
        self._commit('Delete fringe')
        self.set_status(f'Deleted a fringe piece ({n_px} px)')
        self._schedule_view()

    # ------------------------------------------------------------ crop --
    def _apply_crop(self, a, b):
        x0, x1 = sorted((int(round(a[0])), int(round(b[0]))))
        y0, y1 = sorted((int(round(a[1])), int(round(b[1]))))
        x0, x1 = max(0, x0), min(self.grid, x1)
        y0, y1 = max(0, y0), min(self.grid, y1)
        if x1 <= x0 or y1 <= y0:
            return
        names = ['mask', 'gray', 'shot', 'ref', 'ovl']
        self._begin(names)
        # remember what the crop clears so Reset Crop can bring it back
        for name in names:
            arr = self._layer(name)
            if arr is None:
                continue
            if name not in self._uncropped or self._uncropped[name].shape != arr.shape:
                self._uncropped[name] = arr.copy()
            else:
                keep = self._uncropped_keep_region(name, arr.shape)
                self._uncropped[name][keep] = arr[keep]
        new_crop = np.zeros((self.grid, self.grid), np.uint8)
        new_crop[y0:y1, x0:x1] = 255
        self._crop_mask = cv2.bitwise_and(self._crop_mask, new_crop)
        fills = {'mask': 255, 'gray': 0, 'shot': 0, 'ref': 0, 'ovl': 255}
        for name in names:
            arr = self._layer(name)
            if arr is None:
                continue
            px, py = {'ref': self.ref_pos, 'ovl': self.overlay_pos}.get(name, (0, 0))
            out = np.full_like(arr, fills[name])
            ah, aw = arr.shape[:2]
            lx0, lx1 = max(0, x0 - px), min(aw, x1 - px)
            ly0, ly1 = max(0, y0 - py), min(ah, y1 - py)
            if lx1 > lx0 and ly1 > ly0:
                out[ly0:ly1, lx0:lx1] = arr[ly0:ly1, lx0:lx1]
            arr[...] = out
        self._commit('Crop')
        self.set_status(f'Erased outside crop region {x1 - x0}x{y1 - y0}')
        self._schedule_view()

    def _uncropped_keep_region(self, name, shape):
        """Pixels still inside the crop for this layer (in the layer's own coordinates)."""
        px, py = {'ref': self.ref_pos, 'ovl': self.overlay_pos}.get(name, (0, 0))
        h, w = shape[:2]
        keep = np.zeros((h, w), bool)
        ys0, xs0 = max(0, -py), max(0, -px)
        ys1, xs1 = min(h, self.grid - py), min(w, self.grid - px)
        if ys1 > ys0 and xs1 > xs0:
            keep[ys0:ys1, xs0:xs1] = self._crop_mask[ys0 + py:ys1 + py, xs0 + px:xs1 + px] == 255
        return keep

    def reset_crop(self):
        if np.all(self._crop_mask == 255):
            self.set_status('No active crop to reset.')
            return
        names = [n for n in ('mask', 'gray', 'shot', 'ref', 'ovl') if self._layer(n) is not None]
        self._begin(names)
        for name in names:
            arr = self._layer(name)
            back = self._uncropped.get(name)
            if back is None or back.shape != arr.shape:
                continue
            outside = ~self._uncropped_keep_region(name, arr.shape)
            arr[outside] = back[outside]
        self._crop_mask.fill(255)
        self._uncropped = {}
        self._commit('Reset crop')
        self.set_status(f'Crop reset to full {self.grid}x{self.grid} view.')
        self._schedule_view()

    # ----------------------------------------------- connection check ---
    def _ensure_labels(self):
        if self._labels_cache is not None and not self._labels_dirty:
            return self._labels_cache
        bw = (self.mask == 0).astype(np.uint8)
        num, labels = cv2.connectedComponents(bw, connectivity=8)
        idx = np.arange(num, dtype=np.uint32)
        x = (idx * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
        colors = np.stack([(x & np.uint32(0xFF)).astype(np.uint8),
                           ((x >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8),
                           ((x >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)], axis=1)
        c1 = idx % 3; c2 = (idx + 1) % 3
        colors[idx, c1] = np.maximum(colors[idx, c1], 150)
        colors[idx, c2] = np.minimum(colors[idx, c2], 100)
        colors[0] = (255, 255, 255)
        self._labels_cache = (labels, colors)
        self._labels_dirty = False
        return self._labels_cache

    def _connection_check(self, a, b):
        labels, _ = self._ensure_labels()
        trip = np.zeros(self.mask.shape, np.uint8)
        cv2.line(trip, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, 2)
        self.magic2_labels = set(np.unique(labels[trip == 1]).tolist()) - {0}
        self.set_status(f'Connection check: {len(self.magic2_labels)} fringes touched (orange)')
        self._schedule_view()

    # ---------------------------------------------------- link ends ----
    def _link_endpoints(self):
        tgt = self._edit_target()
        if tgt is None:
            self.set_status('No reference binary loaded to link endpoints on')
            return
        name = tgt[0]
        target = self._layer(name)
        try:
            R = max(1, min(300, int(self.link_tol_var.get())))
            try:
                theta_deg = int(self.angle_deg_var.get())
            except Exception:
                theta_deg = 0
            tan_theta = math.tan(math.radians(max(0, min(89, theta_deg)))) if theta_deg > 0 else None
            bw = (target == 0).astype(np.uint8)
            if bw.max() == 0:
                return
            skel = skeletonize(bw > 0).astype(np.uint8)
            nbr = cv2.filter2D(skel, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
            ys, xs = np.where((skel == 1) & (nbr == 2))
            n = len(xs)
            if n < 2:
                self.set_status('No endpoints to link')
                return
            _, labels = cv2.connectedComponents(skel, connectivity=8)
            used = set(); links = 0
            for i in range(n):
                if i in used:
                    continue
                x0, y0 = int(xs[i]), int(ys[i])
                best_j = None; best_d2 = None
                for j in range(n):
                    if j == i or j in used:
                        continue
                    x1, y1 = int(xs[j]), int(ys[j])
                    if abs(x1 - x0) > R or abs(y1 - y0) > R:
                        continue
                    dx = x1 - x0; dy = y1 - y0; d2 = dx * dx + dy * dy
                    if d2 > R * R:
                        continue
                    if tan_theta is not None:
                        if dx == 0 or abs(dy) > tan_theta * abs(dx):
                            continue
                    if labels[y0, x0] > 0 and labels[y0, x0] == labels[y1, x1]:
                        continue
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2; best_j = j
                if best_j is not None:
                    cv2.line(bw, (x0, y0), (int(xs[best_j]), int(ys[best_j])), 1, 1)
                    used.add(i); used.add(best_j); links += 1
            bw = skeletonize(bw > 0).astype(np.uint8)
            self._begin([name])
            target[...] = np.where(bw > 0, 0, 255).astype(np.uint8)
            self._commit('Link endpoints')
            self.set_status(f'Linked {links} endpoint pairs (tol {R}px, angle ±{theta_deg}°)')
            self._schedule_view()
        except Exception:
            messagebox.showerror('Link endpoints', 'Failed to link endpoints on the current mask')

    # ----------------------------------------------- merge / nudge -----
    def merge_overlay(self):
        if self.overlay_mask is None or self.mask is None:
            return
        self._begin(['mask'])
        ox, oy = self.overlay_pos
        oh, ow = self.overlay_mask.shape[:2]
        h, w = self.mask.shape[:2]
        x0 = max(0, ox); y0 = max(0, oy); x1 = min(w, ox + ow); y1 = min(h, oy + oh)
        if x1 > x0 and y1 > y0:
            sx0 = x0 - ox; sy0 = y0 - oy
            ov = self.overlay_mask[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
            self.mask[y0:y1, x0:x1] = cv2.bitwise_and(self.mask[y0:y1, x0:x1], ov)
        self._commit('Merge binaries')
        self.overlay_active = False
        self.overlay_mask = None
        self._uncropped.pop('ovl', None)
        self.btn_merge_overlay.config(state='disabled')
        if self.edit_layer_var.get() == 'ref':
            self.edit_layer_var.set('shot')
        self.set_status('Reference binary merged into the shot binary.')
        self._schedule_view()

    def _nudge_overlay(self, dx, dy):
        moved = False
        if self.overlay_mask is not None:
            self.overlay_pos = (self.overlay_pos[0] + dx, self.overlay_pos[1] + dy)
            moved = True
        if self._ref_img is not None:
            self.ref_pos = (self.ref_pos[0] + dx, self.ref_pos[1] + dy)
            moved = True
        if moved:
            self._invalidate('ovl')
            self._schedule_view()

    def reset_nudge(self):
        moved = False
        if self.overlay_mask is not None:
            self.overlay_pos = self._initial_overlay_pos
            moved = True
        if self._ref_img is not None:
            self.ref_pos = self._initial_ref_pos
            moved = True
        if moved:
            self._invalidate('ovl')
            self._schedule_view()
            self.set_status('Nudge reset to initial positions.')
        else:
            self.set_status('No reference image or binary to reset.')

    # ------------------------------------------------------------ ruler --
    def _apply_ruler_lock(self, anchor, p):
        lock = self.ruler_lock_var.get()
        if lock == 'Horizontal':
            return (p[0], anchor[1])
        if lock == 'Vertical':
            return (anchor[0], p[1])
        return p

    def _ruler_down(self, ix, iy):
        pt = (ix, iy)
        if len(self._ruler_points) == 2:
            for k in (0, 1):
                px, py = self._ruler_points[k]
                if math.hypot(ix - px, iy - py) * self._zoom < 15:
                    self._ruler_drag_idx = k
                    return
            self._ruler_points = [pt]
        elif len(self._ruler_points) == 1:
            self._ruler_points.append(self._apply_ruler_lock(self._ruler_points[0], pt))
        else:
            self._ruler_points = [pt]
        self._redraw_ruler()

    def _ruler_drag(self, ix, iy):
        if self._ruler_drag_idx is not None:
            k = self._ruler_drag_idx
            self._ruler_points[k] = self._apply_ruler_lock(self._ruler_points[1 - k], (ix, iy))
        self._redraw_ruler()

    def _ruler_up(self, ix, iy):
        if self._ruler_drag_idx is not None:
            self._ruler_drag_idx = None
        elif len(self._ruler_points) == 1:
            p0 = self._ruler_points[0]
            if math.hypot(p0[0] - ix, p0[1] - iy) * self._zoom > 3:
                self._ruler_points.append(self._apply_ruler_lock(p0, (ix, iy)))
        self._redraw_ruler()

    def _redraw_ruler(self):
        for item in self._ruler_items:
            try:
                self.canvas.delete(item)
            except Exception:
                pass
        self._ruler_items = []
        if self.mode_var.get() != 'ruler' or not self._ruler_points:
            self.ruler_dist_var.set('Dist: 0.00 px')
            return
        p0 = self._ruler_points[0]
        if len(self._ruler_points) == 2:
            p1 = self._ruler_points[1]
        elif self._last_mouse is not None:
            p1 = self._apply_ruler_lock(p0, self._to_img(*self._last_mouse))
        else:
            p1 = p0
        c0 = self._to_canvas(*p0); c1 = self._to_canvas(*p1)
        self._ruler_items.append(self.canvas.create_line(c0[0], c0[1], c1[0], c1[1], fill='#00ffff', width=2, dash=(4, 4)))
        for cx, cy in (c0, c1):
            self._ruler_items.append(self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill='#ff00ff', outline='white'))
        self.ruler_dist_var.set(f'Dist: {math.hypot(p1[0] - p0[0], p1[1] - p0[1]):.2f} px')

    # =========================================================== render ==
    def _schedule_view(self, fast=False):
        if self._render_id is not None:
            return
        self._render_id = self.after(1 if fast else 8, self._render_full)

    def _render_full(self):
        self._render_id = None
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self._frame_rgb is None or self._frame_rgb.shape[:2] != (ch, cw):
            self._frame_rgb = np.zeros((ch, cw, 3), np.uint8)
        self._render_rect(0, 0, cw, ch, push=False)
        self._push_frame()
        # overlays drawn as canvas items
        self._redraw_ruler()
        if self._tool_points and self._last_mouse is not None:
            self._update_preview(*self._last_mouse)
        self._update_cursor()
        self._update_snap_marker()

    def _push_frame(self):
        self._photo = ImageTk.PhotoImage(Image.fromarray(self._frame_rgb))
        if self._img_item is None:
            self._img_item = self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        else:
            self.canvas.itemconfig(self._img_item, image=self._photo)
        self.canvas.tag_lower(self._img_item)

    def _render_rect(self, cx0, cy0, cx1, cy1, push=True):
        """Recomposite the canvas rectangle [cx0, cx1) x [cy0, cy1)."""
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self._frame_rgb is None or self._frame_rgb.shape[:2] != (ch, cw):
            self._frame_rgb = np.zeros((ch, cw, 3), np.uint8)
            cx0, cy0, cx1, cy1 = 0, 0, cw, ch
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(cw, cx1), min(ch, cy1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        self._frame_rgb[cy0:cy1, cx0:cx1] = self._compose(cx0, cy0, cx1, cy1)
        if push:
            self._push_frame()

    # Zoomed out, each layer is drawn from a cached reduced copy (a "level": 1/2,
    # 1/4, ... of the grid), so a frame never has to shrink a 4096 x 4096 array.
    # Fringe and mask levels keep a pixel lit if any pixel in its block is lit, so
    # 1 px lines never vanish. Levels are rebuilt only when that layer changes.
    def _level(self):
        z = self._zoom
        if z >= 1.0:
            return 0
        return max(0, int(math.ceil(math.log2(1.0 / z) - 1e-9)))

    def _pyr_patch(self, name, x0, y0, x1, y1):
        """Bring cached reduced copies of a binary up to date inside a changed
        rectangle (layer coordinates), so a stroke shows while you draw."""
        base = self._layer(name)
        if base is None:
            return
        ver = self._ver.get(name, 0)
        H, W = base.shape[:2]
        for (n, kind, L), (v, data) in list(self._pyr.items()):
            if n != name or v != ver or kind not in ('ink', 'gray'):
                continue
            f = 2 ** L
            lx0, ly0 = max(0, int(x0) // f), max(0, int(y0) // f)
            lx1, ly1 = min(data.shape[1], -(-int(math.ceil(x1)) // f)), min(data.shape[0], -(-int(math.ceil(y1)) // f))
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            b = base[ly0 * f:ly1 * f, lx0 * f:lx1 * f]
            b = (b == 0) if kind == 'ink' else (b > 0)
            data[ly0:ly1, lx0:lx1] = b.reshape(ly1 - ly0, f, lx1 - lx0, f).any(axis=(1, 3))

    def _pyr_get(self, name, kind, L):
        """Reduced copy of a layer at level L (factor 2**L). kind: 'img', 'ink', 'gray', 'labels'."""
        src_name = 'mask' if kind == 'labels' else name
        ver = self._ver.get(src_name, 0)
        key = (name, kind, L)
        hit = self._pyr.get(key)
        if hit is not None and hit[0] == ver:
            return hit[1]
        if kind == 'labels':
            base = self._ensure_labels()[0]
        else:
            base = self._layer(name)
        if base is None:
            return None
        f = 2 ** L
        H, W = base.shape[:2]
        h, w = max(1, H // f), max(1, W // f)
        if kind == 'img':
            data = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
        else:
            b = base[:h * f, :w * f]
            if kind == 'ink':
                b = b == 0
            elif kind == 'gray':
                b = b > 0
            b = b.reshape(h, f, w, f)
            data = b.max(axis=(1, 3)) if kind == 'labels' else b.any(axis=(1, 3))
        self._pyr[key] = (ver, data)
        return data

    def _sample_layer(self, name, kind, pos, rect, override=None):
        """Draw layer `name` (placed at grid position pos) into canvas rectangle rect.
        Returns uint8 (img), bool (ink/gray) or int32 (labels) of the rect size, or None."""
        cx0, cy0, cx1, cy1 = rect
        Wd, Hd = cx1 - cx0, cy1 - cy0
        z = self._zoom
        L = self._level()
        f = 2 ** L
        px, py = pos
        base = self._layer(name) if kind != 'labels' else (self._ensure_labels()[0] if self.mask is not None else None)
        if base is None:
            return None
        # visible grid window
        gx0 = self._ox + cx0 / z; gy0 = self._oy + cy0 / z
        gx1 = self._ox + cx1 / z; gy1 = self._oy + cy1 / z
        if L == 0:
            data = base
        else:
            data = self._pyr_get(name, kind, L)
        dh, dw = data.shape[:2]
        lx0 = max(0, int(math.floor((gx0 - px) / f)) - 1); ly0 = max(0, int(math.floor((gy0 - py) / f)) - 1)
        lx1 = min(dw, int(math.ceil((gx1 - px) / f)) + 1); ly1 = min(dh, int(math.ceil((gy1 - py) / f)) + 1)
        if lx1 <= lx0 or ly1 <= ly0:
            return None
        sub = data[ly0:ly1, lx0:lx1]
        if L == 0:
            if kind == 'ink':
                sub = sub == 0
            elif kind == 'gray':
                sub = sub > 0
        s = z * f
        tx = (px + lx0 * f - self._ox) * z - cx0
        ty = (py + ly0 * f - self._oy) * z - cy0
        M = np.float32([[s, 0, tx], [0, s, ty]])
        if kind == 'img':
            flags = cv2.INTER_NEAREST if s >= 4 else cv2.INTER_LINEAR
            return cv2.warpAffine(sub, M, (Wd, Hd), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        if kind == 'labels':
            out = cv2.warpAffine(sub.astype(np.float32), M, (Wd, Hd), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return out.astype(np.int32)
        out = cv2.warpAffine(sub.view(np.uint8) if sub.dtype == bool else sub.astype(np.uint8), M, (Wd, Hd),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out > 0

    def _mirror_ink(self, rect):
        if not self.mirror_var.get():
            return None
        try:
            cxm = float(self.mirror_x_var.get().strip())
        except (ValueError, TypeError):
            return None
        cx0, cy0, cx1, cy1 = rect
        Wd, Hd = cx1 - cx0, cy1 - cy0
        z = self._zoom
        L = self._level(); f = 2 ** L
        data = self.mask if L == 0 else self._pyr_get('mask', 'ink', L)
        if data is None:
            return None
        dh, dw = data.shape[:2]
        gx0 = self._ox + cx0 / z; gy0 = self._oy + cy0 / z
        gx1 = self._ox + cx1 / z; gy1 = self._oy + cy1 / z
        lx0 = max(0, int(math.floor(gx0 / f)) - 1); ly0 = max(0, int(math.floor(gy0 / f)) - 1)
        lx1 = min(dw, int(math.ceil(gx1 / f)) + 1); ly1 = min(dh, int(math.ceil(gy1 / f)) + 1)
        if lx1 <= lx0 or ly1 <= ly0:
            return None
        # column x shows the column reflected about the centre line x = cxm
        src = np.floor((2 * cxm - (np.arange(lx0, lx1) + 0.5) * f) / f).astype(np.int64)
        valid = (src >= 0) & (src < dw)
        sub = np.zeros((ly1 - ly0, lx1 - lx0), bool)
        if valid.any():
            cols = data[ly0:ly1, src[valid]]
            sub[:, valid] = (cols == 0) if L == 0 else cols
        s = z * f
        M = np.float32([[s, 0, (lx0 * f - self._ox) * z - cx0], [0, s, (ly0 * f - self._oy) * z - cy0]])
        out = cv2.warpAffine(sub.view(np.uint8), M, (Wd, Hd), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out > 0

    def _grid_inside(self, rect):
        z = self._zoom
        cx0, cy0, cx1, cy1 = rect
        xs = self._ox + (np.arange(cx0, cx1) + 0.5) / z
        ys = self._oy + (np.arange(cy0, cy1) + 0.5) / z
        return ((ys >= 0) & (ys < self.grid))[:, None] & ((xs >= 0) & (xs < self.grid))[None, :]

    def _compose(self, cx0, cy0, cx1, cy1):
        Wd, Hd = cx1 - cx0, cy1 - cy0
        rect = (cx0, cy0, cx1, cy1)
        out = np.empty((Hd, Wd, 3), np.uint8)
        out[:] = OUTSIDE
        inside = self._grid_inside(rect)
        if not inside.any():
            return out
        gray = None
        if self._ref_img is not None and self.show_ref_img.get():
            r = self._sample_layer('ref', 'img', self.ref_pos, rect)
            if r is not None:
                gray = cv2.convertScaleAbs(r, alpha=2.0 ** float(self.ref_brightness_var.get()))
        if self._shot_img is not None and self.show_shot_img.get():
            sh = self._sample_layer('shot', 'img', (0, 0), rect)
            if sh is not None:
                sh = cv2.convertScaleAbs(sh, alpha=2.0 ** float(self.shot_brightness_var.get()))
                a = float(self.shot_opacity_var.get())
                under = gray if gray is not None else np.full((Hd, Wd), 255, np.uint8)
                gray = cv2.addWeighted(sh, a, under, 1.0 - a, 0)
        if gray is None:
            gray = np.full((Hd, Wd), 255, np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if self.show_shot_bin.get() and self.mask is not None:
            ink = self._sample_layer('mask', 'ink', (0, 0), rect)
            mir = self._mirror_ink(rect)
            if mir is not None:
                rgb[mir & ~ink] = RED            # view only, under the black layer
            if ink is not None and ink.any():
                colorize = self.show_components_var.get()
                hl = self.magic2_labels
                if colorize or hl:
                    labels, colors = self._ensure_labels()
                    lab = self._sample_layer('mask', 'labels', (0, 0), rect)
                    if colorize:
                        col = colors[lab]
                        col[lab == 0] = (0, 0, 0)     # drawn since the last relabel
                        rgb[ink] = col[ink]
                    else:
                        rgb[ink] = (0, 0, 0)
                    if hl:
                        hm = np.isin(lab, np.fromiter(hl, np.int64)) & ink
                        rgb[hm] = ORANGE
                else:
                    a = float(self.shot_binary_opacity_var.get())
                    if a >= 0.999:
                        rgb[ink] = 0
                    else:
                        rgb[ink] = (rgb[ink].astype(np.float32) * (1.0 - a)).astype(np.uint8)
        if self.gray_mask is not None:
            g = self._sample_layer('gray', 'gray', (0, 0), rect)
            if g is not None:
                rgb[g] = (128, 128, 128)
        if self.overlay_mask is not None and self.show_ref_bin.get():
            ov = self._sample_layer('ovl', 'ink', self.overlay_pos, rect)
            if ov is not None and ov.any():
                a = float(self.ref_binary_opacity_var.get())
                rgb[ov] = (rgb[ov].astype(np.float32) * (1.0 - a) + np.array(CYAN, np.float32) * a).astype(np.uint8)
        rgb[~inside] = OUTSIDE
        return rgb
