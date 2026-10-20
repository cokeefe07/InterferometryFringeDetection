"""
Orientation- and scale-adaptive fringe ridge detector for laser-plasma interferograms.

Pipeline (all heavy work runs once on the native grid and is cached):

  A. Features (native grid)
     1. despike           remove hot pixels / cosmic hits (median outlier test)
     2. fan filter        one FFT: removes the slowly varying background AND any
                          structure whose wave-vector lies outside +-max_angle of
                          vertical (the vertical streaks next to the target, edge
                          glints).  Fringes up to max_angle from horizontal pass.
     3. orientation       structure tensor of the filtered image -> fringe
                          direction theta(x, y), clamped to +-max_angle
     4. tangent smoothing average ALONG the fringes by following the orientation
                          field (curved streamlines), never across them.  This kills
                          speckle without blurring dense fringes together.
     5. amplitude norm.   divide by the local fringe amplitude, so faint and bright
                          fringes look the same to the ridge detector
     6. ridge strength    second derivative across the fringe at 6 scales; the
                          scale is chosen per pixel (dense fringes near the target
                          get a fine scale, wide fringes a coarse one)
  B. Ridge centre lines   non-maximum suppression across the fringe, on the output
                          grid (2048 or 4096).  The response is smooth, so it is
                          upsampled before NMS and the 4096 lines sit between the
                          native pixel centres.
  C. Clean-up (cheap)     hysteresis threshold, fringe-region mask (plus the
                          optional user-drawn analysis region), junction
                          breaking (so no two fringes are ever 8-connected), short
                          piece removal, orientation-guided gap bridging.
"""
import math
import time
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

F32 = np.float32

DEFAULTS = dict(
    max_angle=70.0,      # deg; largest fringe tilt from horizontal that is kept
    smooth_len=6.0,      # native px; sigma of the averaging along the fringes
    smooth_ratio='auto', # per-pixel cap: sigma <= ratio * local period ('auto' = smooth_len / 12)
    sensitivity=0.5,     # 0..1; higher = keeps fainter ridges
    min_length=30,       # native px; shorter pieces are deleted
    gap_bridge=12,       # native px; largest gap that is bridged
    mask_level=0.85,     # 0..1; how hard short pieces are rejected where fringes are weak
    keep_main_region=False,
    orient_scale=8.0,    # native px; structure tensor integration scale
    fine_bias=-1.0,      # scale-selection bias toward fine scales
    scale_prior=0.0,     # >0: favour ridge scales near 0.225*local period (band factor)
    bg_period=48.0,      # native px; structure coarser than this is background
    despike=True,
    refine_period=False,
    link_rmin=0.0,
    link_rmean=0.1,
    max_dev=40.0,
)

SIGMAS = (0.6, 0.85, 1.2, 1.7, 2.4, 3.4)


# --------------------------------------------------------------------------- #
#  Stage A helpers
# --------------------------------------------------------------------------- #
def despike(I, k=6.0):
    med = cv2.medianBlur(np.ascontiguousarray(I, dtype=F32), 3)
    d = I - med
    mad = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 8)) + F32(1e-6)
    out = I.copy()
    m = np.abs(d) > F32(k) * mad
    out[m] = med[m]
    return out


def fan_filter(I, max_ang=70.0, taper=10.0, pmax=48.0, want_period=False):
    H, W = I.shape
    F = np.fft.rfft2(I - I.mean())
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    kr = np.sqrt(kx * kx + ky * ky)
    alpha = np.degrees(np.arctan2(np.abs(kx), np.abs(ky)))
    a0, a1 = max_ang - taper / 2.0, max_ang + taper / 2.0
    wa = np.clip((a1 - alpha) / (a1 - a0), 0, 1)
    wa = 0.5 - 0.5 * np.cos(np.pi * wa)
    wr = 1.0 - np.exp(-0.5 * (kr * pmax) ** 2)
    Ff = F * (wa * wr)
    out = np.fft.irfft2(Ff, s=(H, W)).astype(F32)
    if not want_period:
        return out
    # exact (spectral) gradient -> local wavenumber k^2 = <|grad f|^2> / <f^2>
    fx = np.fft.irfft2(Ff * (2j * np.pi * kx), s=(H, W)).astype(F32)
    fy = np.fft.irfft2(Ff * (2j * np.pi * ky), s=(H, W)).astype(F32)
    g2 = cv2.GaussianBlur(fx * fx + fy * fy, (0, 0), 8)
    f2 = cv2.GaussianBlur(out * out, (0, 0), 8)
    k = np.sqrt(np.maximum(g2 / (f2 + F32(1e-12)), F32(1e-6)))
    P = (F32(2 * np.pi) / k).astype(F32)
    return out, P


def structure_orientation(f, rho=8.0, pre=0.6, max_ang=70.0, rho_map=None, rhos=(3.0, 5.0, 8.0)):
    g = cv2.GaussianBlur(f, (0, 0), pre) if pre > 0 else f
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    if rho_map is None:
        J11 = cv2.GaussianBlur(gx * gx, (0, 0), rho)
        J22 = cv2.GaussianBlur(gy * gy, (0, 0), rho)
        J12 = cv2.GaussianBlur(gx * gy, (0, 0), rho)
    else:
        # integration scale varies per pixel: interpolate between a few scales
        rhos = [r for r in rhos if r < rho] + [rho]
        gxx, gyy, gxy = gx * gx, gy * gy, gx * gy
        J11 = np.zeros_like(f); J22 = np.zeros_like(f); J12 = np.zeros_like(f)
        rm = np.clip(rho_map, rhos[0], rhos[-1])
        for i, r in enumerate(rhos):
            if i == 0:
                w = np.clip((rhos[1] - rm) / (rhos[1] - rhos[0]), 0, 1) if len(rhos) > 1 else np.ones_like(rm)
            elif i == len(rhos) - 1:
                w = np.clip((rm - rhos[i - 1]) / (rhos[i] - rhos[i - 1]), 0, 1)
            else:
                w = np.clip(np.minimum((rm - rhos[i - 1]) / (rhos[i] - rhos[i - 1]),
                                       (rhos[i + 1] - rm) / (rhos[i + 1] - rhos[i])), 0, 1)
            w = w.astype(F32)
            J11 += w * cv2.GaussianBlur(gxx, (0, 0), r)
            J22 += w * cv2.GaussianBlur(gyy, (0, 0), r)
            J12 += w * cv2.GaussianBlur(gxy, (0, 0), r)
    phi = 0.5 * np.arctan2(2 * J12, J11 - J22)          # gradient direction
    theta = phi + F32(np.pi / 2)                          # fringe direction
    theta = (theta + F32(np.pi / 2)) % F32(np.pi) - F32(np.pi / 2)
    m = F32(np.deg2rad(max_ang))
    theta = np.clip(theta, -m, m).astype(F32)
    coh = (np.sqrt((J11 - J22) ** 2 + 4 * J12 ** 2) / (J11 + J22 + F32(1e-9))).astype(F32)
    return theta, coh


def streamline_smooth(f, theta, sigma, step=1.0, sigma_map=None):
    """Gaussian average of f along the orientation field (follows curvature).
    sigma_map (optional) gives a per-pixel sigma <= sigma."""
    if sigma <= 0:
        return f
    H, W = f.shape
    tx = np.cos(theta).astype(F32)
    ty = np.sin(theta).astype(F32)
    xx, yy = np.meshgrid(np.arange(W, dtype=F32), np.arange(H, dtype=F32))
    J = int(math.ceil(2.5 * sigma / step))
    acc = f.copy()
    wsum = np.ones_like(f) if sigma_map is not None else 1.0
    inv2 = None if sigma_map is None else (F32(-0.5) / (sigma_map * sigma_map)).astype(F32)
    for sgn in (1.0, -1.0):
        X = xx.copy(); Y = yy.copy()
        dx = F32(sgn) * tx; dy = F32(sgn) * ty
        for j in range(1, J + 1):
            # midpoint (RK2) step along the field, keeping a consistent sense
            mx = X + F32(0.5 * step) * dx; my = Y + F32(0.5 * step) * dy
            ntx = cv2.remap(tx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            nty = cv2.remap(ty, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            flip = (ntx * dx + nty * dy) < 0
            ntx[flip] = -ntx[flip]; nty[flip] = -nty[flip]
            X = X + F32(step) * ntx; Y = Y + F32(step) * nty
            dx, dy = ntx, nty
            if inv2 is None:
                w = F32(math.exp(-0.5 * (j * step / sigma) ** 2))
            else:
                w = np.exp(inv2 * F32((j * step) ** 2))
            acc += w * cv2.remap(f, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum = wsum + w
    return (acc / wsum).astype(F32)


def spectral_period(f, win=8.0):
    H, W = f.shape
    F = np.fft.rfft2(f)
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    fx = np.fft.irfft2(F * (2j * np.pi * kx), s=(H, W)).astype(F32)
    fy = np.fft.irfft2(F * (2j * np.pi * ky), s=(H, W)).astype(F32)
    g2 = cv2.GaussianBlur(fx * fx + fy * fy, (0, 0), win)
    f2 = cv2.GaussianBlur(f * f, (0, 0), win)
    k = np.sqrt(np.maximum(g2 / (f2 + F32(1e-12)), F32(1e-6)))
    return (F32(2 * np.pi) / k).astype(F32)


def hess_nn(n, s, c, sig):
    g = cv2.GaussianBlur(n, (0, 0), sig)
    gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * F32(0.25)
    gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * F32(0.25)
    gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * F32(0.25)
    nn = s * s * gxx - F32(2) * s * c * gxy + c * c * gyy
    return (-nn * F32(sig * sig)).astype(F32)


def ridge_scale_select(n, theta, sigmas=SIGMAS, win=6.0, bias=-1.0, sig_hint=None, hint_band=1.6):
    s, c = np.sin(theta).astype(F32), np.cos(theta).astype(F32)
    S = len(sigmas)
    Rs = np.empty((S,) + n.shape, F32)
    Es = np.empty((S,) + n.shape, F32)
    for i, sg in enumerate(sigmas):
        r = hess_nn(n, s, c, sg)
        Rs[i] = r
        Es[i] = cv2.GaussianBlur(r * r, (0, 0), win) * F32(sg ** bias)
        if sig_hint is not None:
            # soft prior: penalise scales far from the period-based hint
            lr = np.log(F32(sg) / sig_hint) / F32(np.log(hint_band))
            Es[i] *= np.exp(-0.5 * lr * lr)
    i = np.argmax(Es, 0)
    im = np.clip(i - 1, 0, S - 1); ip = np.clip(i + 1, 0, S - 1)
    e0 = np.take_along_axis(Es, i[None], 0)[0]
    em = np.take_along_axis(Es, im[None], 0)[0]
    ep = np.take_along_axis(Es, ip[None], 0)[0]
    den = em - 2 * e0 + ep
    ok = (i > 0) & (i < S - 1) & (den < 0)
    off = np.zeros_like(e0)
    off[ok] = 0.5 * (em[ok] - ep[ok]) / den[ok]
    fi = i + np.clip(off, -0.5, 0.5)
    lo = np.clip(np.floor(fi).astype(np.int64), 0, S - 2)
    w = (fi - lo).astype(F32)
    R = (1 - w) * np.take_along_axis(Rs, lo[None], 0)[0] + w * np.take_along_axis(Rs, (lo + 1)[None], 0)[0]
    sig = np.exp(np.interp(fi.ravel(), np.arange(S), np.log(sigmas))).reshape(fi.shape).astype(F32)
    return R.astype(F32), sig


# --------------------------------------------------------------------------- #
#  Stage B: ridge centre lines on the output grid
# --------------------------------------------------------------------------- #
def nms_on_grid(R, theta, up=1, strip=512):
    H, W = R.shape
    if up > 1:
        Ru = cv2.resize(R, (W * up, H * up), interpolation=cv2.INTER_CUBIC)
        c2 = cv2.resize(np.cos(2 * theta), (W * up, H * up), interpolation=cv2.INTER_LINEAR)
        s2 = cv2.resize(np.sin(2 * theta), (W * up, H * up), interpolation=cv2.INTER_LINEAR)
        th = (0.5 * np.arctan2(s2, c2)).astype(F32)
        del c2, s2
    else:
        Ru, th = R, theta
    Hu, Wu = Ru.shape
    nx = (-np.sin(th)).astype(F32)
    ny = np.cos(th).astype(F32)
    keep = np.zeros((Hu, Wu), bool)
    xs = np.arange(Wu, dtype=F32)[None, :]
    for y0 in range(0, Hu, strip):
        y1 = min(Hu, y0 + strip)
        ys = np.arange(y0, y1, dtype=F32)[:, None]
        X = np.broadcast_to(xs, (y1 - y0, Wu))
        Y = np.broadcast_to(ys, (y1 - y0, Wu))
        sx, sy = nx[y0:y1], ny[y0:y1]
        p = cv2.remap(Ru, X + sx, Y + sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.remap(Ru, X - sx, Y - sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        r = Ru[y0:y1]
        keep[y0:y1] = (r > p) & (r >= m) & (r > 0)
    return keep, Ru, th


# --------------------------------------------------------------------------- #
#  Stage C helpers
# --------------------------------------------------------------------------- #
_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def neighbours(b):
    return cv2.filter2D(b.astype(np.uint8), cv2.CV_8U, _K8, borderType=cv2.BORDER_CONSTANT)


def thin(b):
    """Skeletonize only where the line map is thicker than 1 px (2x2 blocks).
    The ridge map is already 1 px almost everywhere, so this is much faster
    than thinning the whole 4096 x 4096 frame."""
    from skimage.morphology import skeletonize
    b = b.astype(bool)
    # 2x2 windows holding 3 or 4 line pixels: blobs and 4-connected staircase corners
    s4 = cv2.filter2D(b.astype(np.uint8), cv2.CV_8U, np.ones((2, 2), np.uint8), anchor=(0, 0),
                      borderType=cv2.BORDER_CONSTANT)
    thick = s4 >= 3
    if not thick.any():
        return b
    zone = cv2.dilate(thick.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(zone.astype(np.uint8), connectivity=8)
    out = b.copy()
    H, W = b.shape
    for i in range(1, n):
        x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        x0, y0 = max(0, x - 3), max(0, y - 3)
        x1, y1 = min(W, x + w + 3), min(H, y + h + 3)
        sk = skeletonize(b[y0:y1, x0:x1])
        reg = lab[y0:y1, x0:x1] == i
        sub = out[y0:y1, x0:x1]
        sub[reg] = sk[reg]
    return out


def break_junctions(b):
    b = b.copy()
    for _ in range(3):
        nb = neighbours(b)
        j = b & (nb >= 3)
        if not j.any():
            break
        b[j] = False
    return b


def break_off_axis(b, theta, max_dev_deg=50.0):
    """Cut the line wherever it runs across the local fringe direction (a jump
    from one fringe to the next).  Uses the direction between each pixel's two
    neighbours, so single-pixel staircase steps are not affected."""
    nb = neighbours(b)
    m = b & (nb == 2)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return b
    pad = np.pad(b, 1)
    vx = np.zeros(len(ys), np.int32); vy = np.zeros(len(ys), np.int32)
    first = np.ones(len(ys), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            hit = pad[ys + 1 + dy, xs + 1 + dx]
            # first neighbour found is subtracted, second added
            sub = hit & first
            add = hit & ~first
            vx[sub] -= dx; vy[sub] -= dy
            vx[add] += dx; vy[add] += dy
            first[sub] = False
    ang = np.arctan2(vy, vx)
    d = np.abs(((ang - theta[ys, xs]) + np.pi / 2) % np.pi - np.pi / 2)
    bad = d > np.deg2rad(max_dev_deg)
    out = b.copy()
    out[ys[bad], xs[bad]] = False
    return out


def remove_short(b, min_len):
    if min_len <= 1:
        return b
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    good = stats[:, cv2.CC_STAT_AREA] >= min_len
    good[0] = False
    return good[lab]


def hysteresis(cand, R, lo, hi):
    weak = cand & (R > lo)
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    strong_lab = np.unique(lab[weak & (R > hi)])
    keep = np.zeros(n, bool); keep[strong_lab] = True; keep[0] = False
    return keep[lab]


def local_period(b, win):
    """Local fringe period from the detected lines: mean distance to the nearest
    line is P/4 for evenly spaced lines."""
    d = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return (4.0 * cv2.blur(d, (int(win) | 1, int(win) | 1))).astype(F32)


def link_gaps(b, theta, period, max_gap, lat_frac=0.35, max_turn_deg=30.0, R=None, r_min=-0.1, r_mean=0.0):
    """Join endpoint pairs that continue each other along the orientation field."""
    H, W = b.shape
    nb = neighbours(b)
    ep = b & (nb == 1)
    ey, ex = np.nonzero(ep)
    if len(ey) < 2:
        return b, 0
    n, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    elab = lab[ey, ex]
    # inward neighbour of each endpoint
    pad = np.pad(b, 1)
    inx = np.zeros(len(ey), F32); iny = np.zeros(len(ey), F32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            hit = pad[ey + 1 + dy, ex + 1 + dx]
            inx[hit] = dx; iny[hit] = dy
    t = theta[ey, ex]
    tx, ty = np.cos(t), np.sin(t)
    sgn = np.where(tx * (-inx) + ty * (-iny) >= 0, 1.0, -1.0)
    ox, oy = tx * sgn, ty * sgn                      # outward tangent
    tree = cKDTree(np.c_[ex, ey])
    pairs = tree.query_pairs(r=max_gap, output_type='ndarray')
    if len(pairs) == 0:
        return b, 0
    i, j = pairs[:, 0], pairs[:, 1]
    keep = elab[i] != elab[j]
    i, j = i[keep], j[keep]
    vx = (ex[j] - ex[i]).astype(F32); vy = (ey[j] - ey[i]).astype(F32)
    dist = np.sqrt(vx * vx + vy * vy) + F32(1e-6)
    # mean tangent of the two ends, oriented along i's outward direction
    mtx = ox[i] - ox[j]; mty = oy[i] - oy[j]
    mn = np.sqrt(mtx * mtx + mty * mty) + 1e-6
    mtx /= mn; mty /= mn
    along = vx * mtx + vy * mty
    lat = np.abs(-vx * mty + vy * mtx)
    P = 0.5 * (period[ey[i], ex[i]] + period[ey[j], ex[j]])
    cos_turn = math.cos(math.radians(max_turn_deg))
    ok = ((ox[i] * vx + oy[i] * vy) > 0) & ((-(ox[j] * vx + oy[j] * vy)) > 0)
    ok &= (-(ox[i] * ox[j] + oy[i] * oy[j])) > cos_turn
    ok &= lat <= np.maximum(1.0, lat_frac * P)
    ok &= along > 0
    if not ok.any():
        return b, 0
    i, j, dist, lat = i[ok], j[ok], dist[ok], lat[ok]
    cost = dist + 4.0 * lat
    order = np.argsort(cost)
    used = np.zeros(len(ey), bool)
    out = b.copy()
    occ = lab  # occupancy by label (original)
    links = 0
    for k in order:
        a, c = i[k], j[k]
        if used[a] or used[c]:
            continue
        x0, y0, x1, y1 = int(ex[a]), int(ey[a]), int(ex[c]), int(ey[c])
        L = int(max(abs(x1 - x0), abs(y1 - y0)))
        if L < 2:
            xs = np.array([x0, x1]); ys = np.array([y0, y1])
        else:
            tt = np.linspace(0, 1, L + 1)
            xs = np.rint(x0 + (x1 - x0) * tt).astype(int); ys = np.rint(y0 + (y1 - y0) * tt).astype(int)
        # reject if the bridge passes next to a third fringe
        la, lc = elab[a], elab[c]
        bad = False
        for xx, yy in zip(xs[1:-1], ys[1:-1]):
            win = occ[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2]
            if np.any((win != 0) & (win != la) & (win != lc)):
                bad = True
                break
        if bad:
            continue
        if R is not None and len(xs) > 2:
            rv = R[ys[1:-1], xs[1:-1]]
            if rv.min() < r_min or rv.mean() < r_mean:
                continue   # the bridge would cross a dark fringe -> different fringe
        out[ys, xs] = True
        used[a] = used[c] = True
        links += 1
    return out, links


# --------------------------------------------------------------------------- #
#  Detector object with stage caching
# --------------------------------------------------------------------------- #
class RidgeDetector:
    def __init__(self, raw, params=None):
        self.raw = raw.astype(F32)
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self._A = None; self._A_key = None
        self._B = {}; self._B_key = None
        self.timings = {}

    # --- stage A ---------------------------------------------------------- #
    def _a_key(self):
        p = self.p
        return (round(p['max_angle'], 3), round(p['smooth_len'], 3), str(p.get('smooth_ratio', 0)), round(p['orient_scale'], 3), int(p.get('orient_iter', 0)), round(p.get('orient_scale2', 5.0), 3), round(p.get('orient_ratio', 0), 3),
                round(p['fine_bias'], 3), round(p.get('scale_prior', 0), 3), round(p['bg_period'], 3), bool(p['despike']))

    def features(self, progress=None):
        key = self._a_key()
        if self._A is not None and self._A_key == key:
            return self._A
        cache = getattr(self, '_A_cache', {})
        if key in cache:
            # recently computed (e.g. auto-optimize or a slider moved back)
            self._A, self._B, self._Pest = cache[key]
            self._A_key = key
            return self._A
        p = self.p
        T = {}
        t = time.time()
        I = despike(self.raw) if p['despike'] else self.raw
        T['despike'] = time.time() - t; t = time.time()
        if progress: progress('Filtering background and streaks')
        f, Pest = fan_filter(I, p['max_angle'], pmax=p['bg_period'], want_period=True)
        T['fan'] = time.time() - t; t = time.time()
        if progress: progress('Estimating fringe orientation')
        rmap = None
        if p.get('orient_ratio', 0) > 0:
            rmap = (F32(p['orient_ratio']) * Pest).astype(F32)
        theta, coh = structure_orientation(f, rho=p['orient_scale'], max_ang=p['max_angle'], rho_map=rmap)
        T['orient'] = time.time() - t; t = time.time()
        if progress: progress('Smoothing along fringes')
        smap = None
        ratio = p.get('smooth_ratio', 0)
        if ratio == 'auto' or ratio is None:
            ratio = p['smooth_len'] / 12.0
        if ratio > 0:
            smap = np.clip(F32(ratio) * Pest, 0.75, max(0.75, p['smooth_len'])).astype(F32)
        ft = streamline_smooth(f, theta, p['smooth_len'], sigma_map=smap)
        if ratio > 0 and p.get('refine_period', True):
            # speckle makes the first period estimate too short in noisy, wide-fringe
            # areas; re-measure on the along-fringe smoothed image and redo
            Pest = np.maximum(Pest, spectral_period(ft))
            smap = np.clip(F32(ratio) * Pest, 0.75, max(0.75, p['smooth_len'])).astype(F32)
            ft = streamline_smooth(f, theta, p['smooth_len'], sigma_map=smap)
        for _ in range(int(p.get('orient_iter', 0))):
            # re-estimate orientation on the along-fringe smoothed image (higher SNR),
            # then smooth again along the improved field
            theta, coh = structure_orientation(ft, rho=p.get('orient_scale2', 5.0), pre=0.0,
                                               max_ang=p['max_angle'])
            ft = streamline_smooth(f, theta, p['smooth_len'], sigma_map=smap)
        self._Pest = Pest
        T['smooth'] = time.time() - t; t = time.time()
        A = np.sqrt(cv2.GaussianBlur(ft * ft, (0, 0), 6.0)) + F32(1e-6)
        n = (ft / A).astype(F32)
        if progress: progress('Measuring ridge strength')
        hint = None
        if p.get('scale_prior', 0) > 0:
            hint = np.clip(F32(0.225) * Pest, SIGMAS[0], SIGMAS[-1]).astype(F32)
        R, sig = ridge_scale_select(n, theta, bias=p['fine_bias'], sig_hint=hint,
                                    hint_band=max(1.05, p.get('scale_prior', 1.6)))
        T['ridge'] = time.time() - t; t = time.time()
        # fringe-region quality: local ridge energy, coherence, illumination
        ped = float(np.percentile(I, 0.5))
        M = cv2.GaussianBlur(I, (0, 0), 8) - F32(ped)
        Mref = float(np.median(M[M > np.percentile(M, 50)]))
        illum = (M / F32(max(Mref, 1e-6))).astype(F32)
        energy = np.sqrt(cv2.GaussianBlur(R * R, (0, 0), 12)).astype(F32)
        cohs = cv2.GaussianBlur(coh, (0, 0), 12)
        T['quality'] = time.time() - t
        self._A = dict(theta=theta, R=R, sig=sig, illum=illum, energy=energy, coh=cohs, flat=n)
        self._A_key = key
        self._B = {}
        self.timings.update(T)
        # keep the previous result too, so toggling between two settings is instant
        old = {k: v for k, v in getattr(self, '_A_cache', {}).items() if k != key}
        old = dict(list(old.items())[-1:])
        old[key] = (self._A, self._B, self._Pest)
        self._A_cache = old
        return self._A

    # --- stage B ---------------------------------------------------------- #
    def candidates(self, up=1, progress=None):
        A = self.features(progress)
        if up in self._B:
            return self._B[up]
        if progress: progress('Locating fringe centre lines')
        t = time.time()
        keep, Ru, th = nms_on_grid(A['R'], A['theta'], up=up)
        self.timings['nms'] = time.time() - t
        self._B[up] = (keep, Ru, th)
        return self._B[up]

    # --- region mask ------------------------------------------------------ #
    def region_mask(self, up=1):
        """Hard mask: only blanks unlit areas (target shadow, outside the probe beam)."""
        A = self.features()
        m = A['illum'] > 0.08
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
        if self.p.get('keep_main_region'):
            q = self.quality_map() > 0.5
            n, lab, stats, _ = cv2.connectedComponentsWithStats((m & q).astype(np.uint8), connectivity=4)
            if n > 1:
                areas = stats[:, cv2.CC_STAT_AREA].copy(); areas[0] = 0
                main = lab == int(np.argmax(areas))
                main = cv2.morphologyEx(main.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
                main = ndi.binary_fill_holes(main)
                m &= main
        if up > 1:
            m = cv2.resize(m.astype(np.uint8), (m.shape[1] * up, m.shape[0] * up), interpolation=cv2.INTER_NEAREST) > 0
        roi = self.p.get('roi')
        if roi is not None:
            # user-drawn analysis region, in native pixel coordinates (x0, y0, x1, y1)
            H, W = m.shape
            ax, ay, bx, by = roi
            x0 = int(np.clip(np.floor(min(ax, bx) * up), 0, W)); x1 = int(np.clip(np.ceil(max(ax, bx) * up), 0, W))
            y0 = int(np.clip(np.floor(min(ay, by) * up), 0, H)); y1 = int(np.clip(np.ceil(max(ay, by) * up), 0, H))
            keep = np.zeros_like(m)
            keep[y0:y1, x0:x1] = True
            m = m & keep
        return m

    def quality_map(self):
        """0..1 fringe quality from orientation coherence (smoothed)."""
        A = self.features()
        lvl = float(self.p['mask_level'])
        lo = 0.2 + 0.3 * lvl
        return np.clip((A['coh'] - lo) / 0.35, 0, 1).astype(F32)

    def quality_length_filter(self, b, up):
        """Short pieces are only trusted where the fringes are coherent: the
        required length grows as the local fringe quality drops."""
        lvl = float(self.p['mask_level'])
        if lvl <= 0:
            return b
        q = self.quality_map()
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
        if n <= 1:
            return b
        ys, xs = np.nonzero(lab)
        L = lab[ys, xs]
        qs = q[np.minimum(ys // up, q.shape[0] - 1), np.minimum(xs // up, q.shape[1] - 1)]
        qmean = np.bincount(L, weights=qs, minlength=n) / np.maximum(1, np.bincount(L, minlength=n))
        length = stats[:, cv2.CC_STAT_AREA]
        need = self.p['min_length'] * up / np.maximum(qmean, 1e-3)
        good = (length >= need) & (qmean > 0.12)
        good[0] = False
        return good[lab]

    # --- stage C ---------------------------------------------------------- #
    def detect(self, up=1, progress=None, return_debug=False):
        p = self.p
        cand, Ru, th = self.candidates(up, progress)
        t = time.time()
        if progress: progress('Cleaning up')
        sens = float(np.clip(p['sensitivity'], 0, 1))
        lo = 0.55 - 0.6 * sens          # sens 0.55 -> 0.22
        hi = lo + 0.18
        mask = self.region_mask(up)
        b = hysteresis(cand & mask, Ru, lo, hi)
        b = thin(b)
        b = break_junctions(b)
        if p.get('max_dev', 0) > 0:
            b = break_off_axis(b, th, p['max_dev'])
        b = remove_short(b, max(3, int(0.3 * p['min_length'] * up)))
        nlinks = 0
        if p['gap_bridge'] > 0:
            per = local_period(b, 31 * up)
            pest = getattr(self, '_Pest', None)
            if pest is not None:
                pe = pest if up == 1 else cv2.resize(pest, (pest.shape[1] * up, pest.shape[0] * up), interpolation=cv2.INTER_LINEAR) * F32(up)
                per = np.minimum(per, pe)
            per = np.clip(per, 3 * up, 40 * up)
            th_u = th
            for frac in (0.5, 1.0):
                b, nl = link_gaps(b, th_u, per, max_gap=max(2, frac * p['gap_bridge'] * up), R=Ru,
                                  r_min=p.get('link_rmin', -0.1), r_mean=p.get('link_rmean', 0.0))
                nlinks += nl
            b = thin(b)
            b = break_junctions(b)
        b = remove_short(b, int(p['min_length'] * up))
        b = self.quality_length_filter(b, up)
        self.timings['cleanup'] = time.time() - t
        self.last_links = nlinks
        if return_debug:
            return b, dict(mask=mask, R=Ru)
        return b


# --------------------------------------------------------------------------- #
#  Auto-optimise
# --------------------------------------------------------------------------- #
def crowding(det, b, up, frac=0.6):
    """Line pixels that have another line closer than frac * local period along
    the fringe normal: spurious ridges between fringes and jumps across them."""
    ys, xs = np.nonzero(b)
    if len(ys) == 0:
        return 0
    A = det.features()
    th = A['theta'][ys // up, xs // up]
    P = det._Pest[ys // up, xs // up] * up
    nx, ny = -np.sin(th), np.cos(th)
    H, W = b.shape
    lim = np.maximum(2.0, frac * P)
    crowded = np.zeros(len(ys), bool)
    tmax = int(np.ceil(lim.max()))
    for t in range(2, tmax + 1):
        act = (lim >= t) & ~crowded
        if not act.any():
            break
        for sg in (1, -1):
            yy = np.rint(ys[act] + sg * t * ny[act]).astype(np.int64)
            xx = np.rint(xs[act] + sg * t * nx[act]).astype(np.int64)
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            hit = np.zeros(ok.shape, bool)
            hit[ok] = b[yy[ok], xx[ok]]
            idx = np.nonzero(act)[0][hit]
            crowded[idx] = True
    return int(crowded.sum())


def quality_score(det, b, up, long_len=100.0, end_cost=60.0, crowd_cost=5.0):
    """Unsupervised score of a detection (native-pixel units).

    Rewards length in long fringes; charges each broken end inside the lit
    region (a fringe that stops where fringes continue) and every pixel of
    short, stray pieces. Ends at the image border or at a shadow edge are free.
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    lens = st[1:, cv2.CC_STAT_AREA] / float(up)
    good = float(lens[lens >= long_len].sum())
    short = float(lens[lens < long_len].sum())
    nb = neighbours(b)
    ep = b & (nb == 1)
    mg = int(40 * up)
    lit = det.region_mask(up)
    inner = cv2.erode(lit.astype(np.uint8), np.ones((2 * mg + 1, 2 * mg + 1), np.uint8)) > 0
    inner[:mg, :] = False; inner[-mg:, :] = False; inner[:, :mg] = False; inner[:, -mg:] = False
    n_end = int((ep & inner).sum())
    crowd = crowding(det, b, up) / float(up)
    score = good - end_cost * n_end - short - crowd_cost * crowd
    return score, dict(long=good, short=short, ends=n_end, crowded=crowd)


AUTO_GRID = dict(smooth_len=(3.0, 6.0),
                 sensitivity=(0.35, 0.45, 0.55, 0.65, 0.8),
                 mask_level=(0.5, 0.85))


def auto_optimize(det, up=1, base=None, progress=None, grid=AUTO_GRID, eval_up=None):
    """Grid search over the settings that matter most, scored by quality_score.
    Candidates are scored on the working grid (merges only show up properly on
    the grid that will be used). Returns (best_params, binary, report)."""
    base = dict(base or {})
    if eval_up is None:
        eval_up = up
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in
              np.array(np.meshgrid(*[grid[k] for k in keys], indexing='ij')).reshape(len(keys), -1).T]
    # group by the expensive stage-A parameter so features are computed once per value
    combos.sort(key=lambda c: c['smooth_len'])
    best = None
    for i, c in enumerate(combos):
        p = dict(base); p.update({k: float(v) for k, v in c.items()})
        det.p.update(p)
        if progress:
            progress('Auto-optimize: trying setting %d of %d' % (i + 1, len(combos)))
        b = det.detect(up=eval_up)
        s, info = quality_score(det, b, eval_up)
        if best is None or s > best[0]:
            best = (s, p, info)
    det.p.update(best[1])
    if progress:
        progress('Auto-optimize: applying best setting')
    b = det.detect(up=up)   # cached stages make this cheap
    p = best[1]
    report = ('Auto-optimize done: sensitivity %.2f, smoothing %.1f, noise rejection %.2f'
              % (p['sensitivity'], p['smooth_len'], p['mask_level']))
    return {k: p[k] for k in p if k in DEFAULTS}, b, report
