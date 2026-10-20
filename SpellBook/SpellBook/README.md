# SpellBook — Fringe Detection & Overlay App

SpellBook is a Tkinter-based desktop tool for:
1. Preprocessing (even illumination / contrast enhancement)
2. Automatic fringe (line) detection & skeletonization
3. Manual mask touch‑up (add/remove/link endpoints)

> **Note:** A detailed user manual is available in `help.html`.

---
## 1. Repository Layout

| Path | Purpose |
|------|---------|
| `SpellBook.py` | Entry point launching the GUI (Detection, Editor tabs). |
| `fringe_detection/` | Processing helpers: shading pipeline, binarization & oriented opening utilities. |
| `tabs/fringe_editor.py` | Interactive binary mask editor (paint + link endpoints). |
| `mixins/viewport_rendering.py` | Shared viewport zoom/pan helpers (legacy mixin). |
| `requirements.txt` | Python dependencies. |

---
## 2. Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python SpellBook.py
```

macOS / Linux (bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python SpellBook.py
```

If wheel installation fails (e.g. on Apple Silicon), use a Conda/Miniforge environment.

---
## 3. Tutorial Walkthrough

### Step 0: Prepare Your Images
Place raw input images in a convenient folder. Supported formats: PNG / TIFF / JPG. High bit‑depth TIFFs are internally normalized to 8‑bit for display.

### Step 1: Launch & Load (Detection Tab)
1. Click `Browse & Load` to pick an image.
2. Adjust preprocessing sliders:
   - Blur σ: mild Gaussian smoothing for noise reduction.
   - CLAHE clip & tile: local contrast expansion; high clip increases contrast, tile defines grid size.
3. Original overlay (optional): blend some of the original intensity back into processed views for context.

The center viewer shows the detected fringe skeleton overlayed (green/colored) on the processed background.

Zoom: mouse wheel. Pan: right‑click drag. Status bar shows current zoom.

### Step 2: Tune Fringe Detection
Right panel sliders:
| Slider | Effect |
|--------|--------|
| Kernel length | Line structuring element length for oriented opening (line extraction). |
| Kernel thickness | Line thickness assumption. |
| Angle ± / Angle step | Angular sweep around horizontal for oriented opening (coverage vs. speed). |
| Dilate px | Thickens detected ridges before skeletonization. |
| Min area | Removes small specks before skeletonizing. |
| Prune spurs | Removes small branches from the skeletonized fringes. |
| Fill loops | Closes small holes within fringes. |
| Hump Width | Flattens fringes by removing "humps" (0 = off). |
| Background opacity | Adjusts the visibility of the original image behind the detected fringes. |

After adjustments the overlay viewer updates automatically (debounced ~180 ms).

### Step 3: Save Automatic Result
Click `Save Fringes as Binary` in the  to export the current binary mask (0 = fringe, 255 = background).

### Step 4: Fine Editing (Editor Tab)
Switch to `Editor`:
1. `Open Shot Binary` / `Open Reference Binary` – load saved masks OR create one from an image.
2. `Load Shot Image` / `Load Ref Image` – load grayscale raw images for context behind the fringes.
3. **Alignment & Cropping**:
   - Use Arrow Keys to nudge the Reference Image and Binary.
   - `Reset Nudge` restores original loaded positions.
   - `Crop View`: Drag a box to crop the view to 2048x2048, preserving your edits. `Reset Crop` restores the full view.
4. **Drawing Tools & Modes**:
   - `Add/Remove Black`: Draw or erase binary fringe lines.
   - `Draw Line (1px)`: Draw straight lines.
   - `Draw Spline (1px)`: Draw curves (Left-click to add points, Ctrl+Left-click to finish).
   - `Add/Remove Mask (Gray)`: Paint a gray mask to exclude regions.
5. **Mouse & Shortcuts**:
   - Left drag: paint / use tool.
   - Ctrl + wheel: change brush radius.
   - Wheel: zoom.
   - Right drag: pan.
   - `Ctrl + Z`: Undo.
6. **Analysis Tools**:
   - `Ruler`: Measure pixels; can lock to Horizontal or Vertical directions.
   - `Link endpoints`: Connect nearby skeleton endpoints within tolerance & angular constraint.
   - `Magic2 Line`: Draws an invisible vertical line to highlight touching fringes (useful for splitting).
   - `Color comps`: Pseudo‑colors connected components.
   - `Merge Binaries`: Combine another binary mask with the current one.
7. **Visualization Adjustments**:
   - Adjust `Shot Brightness`, `Ref Brightness`, `Shot Opacity`, `Shot Binary Opacity`, and `Ref Binary Opacity` sliders for better visibility.

No mask is auto‑loaded into the Editor; you decide when to load or import one.

### Step 5: Iterate & Export
Refine, then `Save As…` in the Editor for a cleaned fringe mask. Use saved masks for downstream analysis or comparison.

---
## 4. Design Notes
| Aspect | Choice |
|--------|--------|
| Zoom & Pan | Smooth inspection with mouse wheel and right-click drag. |
| Skeletonization | Uses `skimage.morphology.skeletonize` on prefiltered binary. |
| Oriented opening | Sweeps discrete angles (± range, given step) with line structuring elements for ridge isolation. |
| Endpoint linking | Brute-force within radius + angle gate + component separation. |
| Performance | Debounced slider changes; uses integer structuring elements. |

---
## 5. Building a Standalone Executable (Windows)
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --name "SpellBook" SpellBook.py
```
Binary appears in `dist/SpellBook.exe`.

GitHub Actions can be configured to run this on tag push (e.g. `v1.2.0`).

---
## 6. Troubleshooting
| Issue | Fix |
|-------|-----|
| Wheel zoom not working | Ensure window focus; on Linux use `<Button-4>/<Button-5>` events. |
| Missing DLL (OpenCV) | Reinstall with `pip install --force-reinstall opencv-python`. |
| Slow large images | Reduce Blur σ & CLAHE tile; disable Color comps during editing. |
| No fringes detected | Increase kernel length, lower Min area, adjust Angle ±. |

---
## 7. Contributing
1. Fork & branch: `git checkout -b feature/xyz`.
2. Keep UI changes minimal per commit.
3. Run lint/tests (add if missing) before PR.

---
## 8. License & Contact
Add `LICENSE` (MIT recommended) and maintainer contact/email here.

---
## 9. At-a-Glance Commands
```powershell
# Setup
python -m venv .venv; ./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Run
python SpellBook.py

# Build executable
pyinstaller --onefile --name SpellBook SpellBook.py
```

---
## 10. FAQ
**Q: Why isn’t the Editor auto-filled?** Manual control prevents accidental edits; load explicitly.
**Q: Units of link tolerance?** Pixels in original image coordinates.

---
Happy detecting and editing! Feel free to open issues for feature requests.
