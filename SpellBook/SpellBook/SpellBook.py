import tkinter as tk
from tkinter import ttk, messagebox

from tabs.detection_tab import DetectionTabFrame


class EvenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SpellBook — Fringe Detection')
        self.geometry('1100x700')
        try:
            self.after(100, self.lift)
            self.after(120, lambda: self.attributes('-topmost', True))
            self.after(700, lambda: self.attributes('-topmost', False))
        except Exception:
            pass

        # --- Header strip: grid switch lives in the top-right corner ---
        # The switch is global because the Detection and Editor tabs have to
        # agree on a grid; a mismatch is refused when applying an edited mask.
        header = ttk.Frame(self)
        header.pack(side='top', fill='x', padx=6, pady=(4, 0))

        self.grid_factor_var = tk.IntVar(value=1)
        ttk.Radiobutton(header, text='4096', variable=self.grid_factor_var,
                        value=2, command=self._on_grid_change).pack(side='right')
        ttk.Radiobutton(header, text='2048', variable=self.grid_factor_var,
                        value=1, command=self._on_grid_change).pack(side='right', padx=(0, 4))
        ttk.Label(header, text='Working grid:').pack(side='right', padx=(0, 6))

        # Notebook tabs: Overlay, Detection, Editor
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Optional status bar at bottom
        self.statusbar = ttk.Label(self, text='Ready')
        self.statusbar.pack(fill='x', side='bottom')

        # Overlay tab

        # Detection tab
        try:
            self._detection_frame = DetectionTabFrame(self.notebook, status_callback=self.set_status)
            self.notebook.add(self._detection_frame, text='Detection')
        except Exception as e:
            self._detection_frame = None
            try:
                self.set_status('Failed to load Detection tab')
            except Exception:
                pass
            # Log exception to console for debugging
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass

        # Editor tab
        editor_tab = ttk.Frame(self.notebook)
        self.notebook.add(editor_tab, text='Editor')

        from tabs.fringe_editor import FringeEditorFrame
        self._editor_frame = None
        if FringeEditorFrame is not None:
            def on_apply(mask, _bg):
                try:
                    if self._detection_frame is not None:
                        self._detection_frame.apply_editor_mask(mask)
                        self.set_status('Applied edited mask from Editor')
                except Exception:
                    pass

            def on_close():
                pass

            self._editor_frame = FringeEditorFrame(editor_tab, on_apply=on_apply, on_close=on_close)
            self._editor_frame.pack(fill='both', expand=True)

        # Detection -> Editor hand-off (the Detection tab saves first, then calls this)
        self._editor_tab = editor_tab
        if self._detection_frame is not None and self._editor_frame is not None:
            self._detection_frame.send_to_editor_cb = self._send_to_editor

    def _send_to_editor(self, binary, shot_img, saved_path, name):
        ok = self._editor_frame.receive_from_detection(binary, shot_img, saved_path, name)
        if ok:
            self.notebook.select(self._editor_tab)
        return ok

    def _on_grid_change(self):
        """Apply the grid switch to both tabs, keeping them in lockstep."""
        factor = int(self.grid_factor_var.get())

        # The Editor asks for confirmation when going down, because shrinking a
        # traced skeleton fuses fringes. If the user cancels there, roll the
        # radio button back and leave both tabs untouched.
        if self._editor_frame is not None:
            try:
                ok = self._editor_frame.set_grid_factor(factor)
            except Exception:
                import traceback
                traceback.print_exc()
                ok = False
            if ok is False:
                prev = 1 if factor == 2 else 2
                self.grid_factor_var.set(prev)
                return

        if self._detection_frame is not None:
            try:
                self._detection_frame.set_grid_factor(factor)
            except Exception:
                import traceback
                traceback.print_exc()

        px = 2048 * factor
        self.set_status(f'Working grid: {px}x{px}')

    def set_status(self, txt):
        try:
            self.statusbar.config(text=txt)
        except Exception:
            pass

    def on_close(self):
        try:
            self.destroy()
        except Exception:
            pass


def main():
    app = EvenApp()
    app.mainloop()


if __name__ == '__main__':
    main()