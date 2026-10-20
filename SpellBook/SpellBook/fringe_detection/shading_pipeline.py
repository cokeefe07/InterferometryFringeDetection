import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def read_gray(path, upsample=1, robust=False):
    """Read an image as uint8 grayscale, robust to 16-bit TIFFs and color inputs.

    - Loads with IMREAD_UNCHANGED to preserve bit depth.
    - If color, converts to grayscale.
    - Optionally upsamples by an integer factor *before* the 8-bit conversion,
      so that interpolation happens on the full dynamic range rather than on
      already-crushed data. Uses INTER_CUBIC: the interferogram is a smooth
      band-limited signal, so cubic places fringe centres between the original
      pixel centres. (Nearest would just replicate pixels and gain nothing.)
    - Normalizes to uint8. With robust=False this is the original min/max
      rescale. With robust=True it clips at the 0.1/99.9 percentiles first, so
      a single hot pixel or a saturated spot can't compress the rest of the
      image into a narrow band.

    Args:
        path: image path
        upsample: integer factor (1 = native, 2 = 2048 -> 4096)
        robust: use percentile clipping instead of min/max
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # Convert color to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Work in float32 from here so the upsample sees the full input range
    work = img.astype(np.float32)

    upsample = max(1, int(upsample))
    if upsample > 1:
        work = cv2.resize(work, None, fx=upsample, fy=upsample,
                          interpolation=cv2.INTER_CUBIC)

    if robust:
        lo, hi = np.percentile(work, (0.1, 99.9))
    else:
        lo, hi = float(np.min(work)), float(np.max(work))

    if hi <= lo:
        return np.zeros(work.shape, dtype=np.uint8)

    out = (work - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def pipeline_shading_sauvola(img_gray, sigma=35.0, clip=2.5, tile=8, win=31, k=0.20, post_open=1):
    """Return (flat, enh, binary) where binary is uint8 {0,255}."""
    bg = cv2.GaussianBlur(img_gray, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.divide(img_gray, bg, scale=255)
    tile = max(2, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    enh = clahe.apply(flat)
    win = int(win) if int(win) % 2 == 1 else int(win) + 1
    thv = threshold_sauvola(enh, window_size=win, k=float(k))
    binary = (enh > thv).astype(np.uint8) * 255
    if post_open > 0:
        ksz = max(1, int(post_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((ksz, ksz), np.uint8))
    return flat, enh, binary