"""Wonder Woman inspired collectible figure - procedural SDF sculpt.

Original geometry, built entirely from analytic signed distance fields:
a heroic-proportion female figure (~7.5 heads) in the classic power pose -
hands on hips, chin up, long wavy hair, cape sweeping back to the base.

All coordinates are millimetres.  The soles stand on z = 0, the display
base sits below that, the character faces +Y and +Z is up.
"""
import numpy as np

from sdf_lib import (F, ln, smin, smax, union, subtract, rotmat, sphere,
                     ellipsoid, capsule, round_cone, box, torus, cylinder,
                     slab, seg2, star5, proj, sphere_proj, sphere_uv,
                     emboss, engrave)

# --------------------------------------------------------------- skeleton
Z_KNEE = 39.0
Z_HIP = 78.0
Z_WAIST = 92.0
Z_BUST = 106.0
Z_SHOULDER = 115.0
Z_HEAD = 136.8
Z_CROWN = 148.0

BASE_R = 26.0
BASE_H = 4.0
BASE_Y = -3.0

# generous margin so the surface never touches the voxel grid wall
BOUNDS = ((-32.0, 32.0), (-34.0, 26.0), (-BASE_H - 3.0, 153.0))

SIDES = (-1.0, 1.0)

# the head is turned slightly to her left, which keeps the pose from
# reading as a mannequin; everything head-related is evaluated in this frame
FACE_C = (0.0, 1.2, 136.8)     # decal projection centre for the face
FACE_R = 9.0

HEAD_PIVOT = np.array([0.0, 0.5, 120.0], dtype=F)
HEAD_ROT = rotmat(rx=-3.0, rz=9.0)


def fp(P):
    return sphere_proj(P, FACE_C, FACE_R)


def fuv(x, z):
    return sphere_uv(x, z, FACE_C, FACE_R)


def head_frame(P):
    return (P - HEAD_PIVOT) @ HEAD_ROT + HEAD_PIVOT


# ------------------------------------------------------------------ torso
def _torso(P):
    d = ellipsoid(P, (0, -0.5, Z_HIP), (13.9, 9.0, 11.0))              # pelvis
    d = smin(d, ellipsoid(P, (0, -5.8, 80.0), (10.8, 5.4, 7.2)), 4.0)  # seat
    d = smin(d, ellipsoid(P, (0, -0.6, Z_WAIST), (9.3, 7.0, 8.2)), 4.5)  # waist
    d = smin(d, ellipsoid(P, (0, -0.6, 105.5), (12.4, 8.3, 11.6)), 4.5)  # ribs
    d = smin(d, ellipsoid(P, (0, -1.0, 113.8), (11.4, 7.2, 5.6)), 4.0)   # pecs
    for s in SIDES:                                                     # bust
        d = smin(d, sphere(P, (s * 5.1, 5.5, Z_BUST), 4.85), 3.2)
    for s in SIDES:                                                     # deltoids
        d = smin(d, sphere(P, (s * 14.0, 0.0, Z_SHOULDER), 5.6), 3.4)
    d = smin(d, round_cone(P, (0, 1.6, 128.6), (0, 0.0, 116.0), 3.6, 5.9), 3.0)
    return d


# ------------------------------------------------------------------- head
def _head(P):
    d = ellipsoid(P, (0, 1.2, Z_HEAD), (7.4, 8.8, 9.7))
    d = smax(d, round_cone(P, (0, 1.0, 141.5), (0, 4.4, 128.2), 9.2, 3.5), 2.2)
    d = smin(d, sphere(P, (0, 7.3, 130.5), 2.3), 2.6)                   # chin
    for s in SIDES:
        d = smin(d, ellipsoid(P, (s * 3.8, 4.6, 134.8), (2.4, 2.2, 2.8)), 3.4)
        d = smin(d, ellipsoid(P, (s * 7.4, 0.4, 135.6), (1.0, 1.9, 2.4)), 1.0)
        d = smin(d, round_cone(P, (s * 1.2, 8.3, 139.6),
                               (s * 4.6, 7.2, 139.9), 0.55, 0.3), 2.2)   # brow ridge
    # nose
    d = smin(d, round_cone(P, (0, 8.3, 140.8), (0, 10.2, 135.4), 0.45, 0.85), 1.1)
    for s in SIDES:
        d = smin(d, sphere(P, (s * 1.25, 9.5, 135.0), 0.62), 0.6)         # nostril wing
    # lips
    d = smin(d, ellipsoid(P, (0, 8.9, 132.2), (2.0, 0.6, 1.15)), 1.3)   # lips
    return d


def _face_cuts(P, d):
    """Eyes are a ball set into an almond socket with a carved lid crease -
    raised eyelids read as floating bars at this scale, carved ones do not.
    Every decal is measured against a frozen copy of the field so the small
    approximation error of one cannot stack onto the next."""
    for s in SIDES:
        d = smax(d, -ellipsoid(P, (s * 3.0, 9.5, 137.3), (1.6, 1.5, 1.0)), 0.35)
        d = np.minimum(d, sphere(P, (s * 3.0, 8.05, 137.3), 1.25))
    base = d.copy()
    front = -(P[:, 1] - F(4.0))
    q = fp(P)                       # arc-length coordinates over the skull
    cut = np.full_like(d, 1e3)
    for s in SIDES:
        cut = np.minimum(cut, engrave(base, seg2(
            q, fuv(s * 1.2, 138.6), fuv(s * 4.6, 138.25), 0.2, 0.14), 0.35, front))
        cut = np.minimum(cut, engrave(base, seg2(
            q, fuv(s * 1.4, 136.2), fuv(s * 4.4, 136.4), 0.16, 0.12), 0.25, front))
    cut = np.minimum(cut, engrave(base, seg2(
        q, fuv(-1.55, 132.2), fuv(1.55, 132.2), 0.24), 0.55, front))
    d = smax(d, -cut, 0.2)
    for s in SIDES:
        d = smax(d, -sphere(P, (s * 1.1, 9.9, 134.9), 0.45), 0.2)         # nostril
    return d


# ------------------------------------------------------------------- arms
def _arms(P):
    d = None
    for s in SIDES:
        S = (s * 14.0, 0.0, Z_SHOULDER)
        E = (s * 22.8, -5.8, 96.0)
        W = (s * 15.6, 2.2, 86.4)
        a = round_cone(P, S, E, 4.3, 3.05)
        a = smin(a, round_cone(P, E, W, 3.2, 2.5), 2.2)
        R = rotmat(rx=20, rz=s * -16)
        a = smin(a, box(P, (s * 14.0, 4.0, 82.4), (1.35, 2.5, 3.3), 1.0, R), 1.5)
        a = smin(a, capsule(P, (s * 15.4, 1.6, 84.4),
                            (s * 14.8, 5.2, 81.2), 1.1), 1.1)           # thumb
        d = a if d is None else np.minimum(d, a)
    return d


def _hand_cuts(P, d):
    for s in SIDES:
        for t in (-1.0, 0.0, 1.0):
            a = (s * (14.1 + t * 0.85), 6.2, 80.2 - abs(t) * 0.3)
            b = (s * (13.6 + t * 0.85), 2.2, 79.2 - abs(t) * 0.3)
            d = smax(d, -engrave(d, capsule(P, a, b, 0.32), 0.7), 0.26)
    return d


def _bracers(P):
    d = None
    for s in SIDES:
        b = round_cone(P, (s * 20.9, -3.8, 93.2), (s * 16.1, 1.5, 87.1), 4.3, 3.5)
        d = b if d is None else np.minimum(d, b)
    return d


# ------------------------------------------------------------------- legs
def _legs(P):
    d = None
    for s in SIDES:
        H = (s * 7.0, 0.0, 74.0)
        K = (s * 9.6, 1.4, Z_KNEE)
        A = (s * 11.0, 0.6, 8.4)
        l = round_cone(P, H, K, 6.9, 4.5)                               # thigh
        l = smin(l, round_cone(P, K, A, 4.4, 2.85), 2.6)                # shin
        l = smin(l, ellipsoid(P, (s * 10.5, -2.6, 29.0), (3.7, 4.0, 7.4)), 3.2)
        l = smin(l, ellipsoid(P, (s * 9.3, 1.4, 43.5), (4.3, 4.5, 4.4)), 3.0)
        R = rotmat(rz=s * -6)
        l = smin(l, box(P, (s * 11.1, 2.4, 3.1), (2.7, 5.4, 2.2), 1.0, R), 1.9)
        l = smin(l, round_cone(P, (s * 11.1, 3.2, 3.3),
                               (s * 10.9, 9.0, 2.3), 2.7, 1.7), 1.5)    # toe
        d = l if d is None else np.minimum(d, l)
    return d


def _boot_top(P):
    """Rim of the over-the-knee boot: peaks at the front, dips at the back."""
    t = np.clip((P[:, 1] - F(1.4)) / F(5.4), -1.0, 1.0)
    return F(44.0) + F(4.6) * t


def _boots(P, legs):
    d = np.maximum(legs - F(0.95), P[:, 2] - _boot_top(P))
    return smax(d, -(P[:, 2] + F(BASE_H + 4.0)), 0.4)


# ------------------------------------------------------------------- hair
def _hair(P):
    cap = ellipsoid(P, (0, -0.3, 138.2), (8.5, 9.4, 9.3))
    cap = smin(cap, sphere(P, (0, -3.0, 140.5), 7.4), 3.0)              # crown
    # a box-shaped opening gives a proper hairline instead of a bald dome
    face = box(P, (0, 12.2, 134.2), (4.7, 5.5, 6.4), 2.4)
    cap = smax(cap, -face, 1.2)
    d = cap
    d = smin(d, ellipsoid(P, (0, -6.8, 126.0), (8.7, 6.4, 15.5)), 3.2)  # back mass
    d = smin(d, round_cone(P, (0, -8.2, 116.0), (0, -6.0, 101.5), 7.2, 3.4), 3.0)
    for s in SIDES:                                                     # locks
        d = smin(d, round_cone(P, (s * 6.6, -3.6, 133.0),
                               (s * 8.8, -3.4, 118.0), 3.4, 3.0), 2.5)
        d = smin(d, round_cone(P, (s * 8.8, -3.4, 118.0),
                               (s * 8.4, -0.2, 108.0), 3.0, 2.3), 2.4)
        d = smin(d, round_cone(P, (s * 8.4, -0.2, 108.0),
                               (s * 7.2, 2.0, 100.0), 2.3, 1.1), 2.0)
        d = smin(d, round_cone(P, (s * 8.2, -6.0, 126.0),
                               (s * 10.2, -4.4, 110.0), 2.6, 1.9), 2.4)
    return d


def _hair_strands(P, d):
    """Grooves that break the hair into locks: great circles over the crown
    plus straight strands running down the back."""
    base = d.copy()
    cut = np.full_like(d, 1e3)
    # the rings must stay inside the hair - behind the face plane or above the
    # hairline - otherwise they comb grooves straight across the brow
    hair_only = np.minimum(P[:, 1] - F(3.5), F(143.2) - P[:, 2])
    for a in (-40.0, -27.0, -14.0, 0.0, 14.0, 27.0, 40.0):
        g = torus(P, (0, -0.3, 138.4), 9.7, 0.42, rotmat(ry=90, rz=a))
        g = np.maximum(np.maximum(g, F(129.0) - P[:, 2]), hair_only)
        cut = np.minimum(cut, engrave(base, g, 0.6))
    for s in SIDES:
        for x0, x1, z0, z1 in ((2.4, 4.6, 140.0, 114.0),
                               (5.4, 8.2, 138.0, 108.0),
                               (7.6, 9.6, 133.0, 104.0)):
            g = capsule(P, (s * x0, -4.0, z0), (s * x1, -9.5, z1), 0.4)
            cut = np.minimum(cut, engrave(base, g, 0.65))
    return smax(d, -cut, 0.45)


# ------------------------------------------------------------------- cape
def _cape(P):
    """A cape as a swept shell: radius grows towards the hem, ripples with a
    couple of harmonics for folds, and the angular span widens as it falls, so
    it springs from between the shoulder blades and flares out at the ground."""
    x = P[:, 0]
    y = P[:, 1] - F(1.0)
    z = P[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(x, y)                    # 0 = front, +-pi = straight back
    t = np.clip((F(114.0) - z) / F(112.0), 0.0, 1.0)

    R = F(13.0) + F(15.0) * t ** 1.6
    R = (R + F(2.4) * np.sin(F(9.0) * th + F(0.5)) * t ** 1.2
         + F(1.0) * np.sin(F(15.0) * th - F(1.0)) * t * t
         + F(1.8) * np.sin(th) * t)          # a touch of asymmetry
    # narrow at the shoulders, wide at the hem
    lim = np.radians(F(112.0) + F(42.0) * np.clip((z - F(66.0)) / F(46.0), 0.0, 1.0))
    edge = np.clip((np.abs(th) - lim) / F(0.45), 0.0, 1.0)
    # the upper corners and the top hem tuck back into the torso so the cape
    # grows out of the shoulder blades; lower down the edges hang free
    tuck = np.clip((z - F(86.0)) / F(16.0), 0.0, 1.0)
    top = np.clip((F(112.0) - z) / F(9.0), 0.0, 1.0)
    R = R * (F(1.0) - F(0.42) * (F(1.0) - edge) * tuck) * (F(0.55) + F(0.45) * top)

    half = (F(1.95) - F(0.5) * t) * F(0.5)
    d = np.abs(r - R) - half
    d = smax(d, -(np.abs(th) - lim) * np.minimum(r, F(14.0)), 0.9)
    lift = (1.0 - np.clip((np.abs(th) - lim) / F(0.42), 0.0, 1.0)) ** 2
    hem = F(0.5) + F(26.0) * lift - F(4.0) * np.sin(F(9.0) * th + F(0.5))
    d = smax(d, hem - z, 0.9)
    d = smax(d, z - F(115.0), 0.8)

    coll = torus(P, (0, -0.8, 117.6), 6.4, 1.45, rotmat(rx=12))
    coll = np.maximum(coll, P[:, 1] - F(0.6))
    return np.minimum(d, coll)


# ------------------------------------------------------------------ lasso
def _lasso(P):
    R = rotmat(rx=90, rz=-12, ry=8)
    d = torus(P, (15.4, 5.0, 75.4), 4.9, 0.9, R)
    d = np.minimum(d, torus(P, (15.8, 3.6, 75.8), 4.1, 0.78, R))
    d = np.minimum(d, capsule(P, (14.6, 4.2, 80.2), (13.8, 2.8, 85.2), 0.72))
    return d


# ------------------------------------------------------------------- base
def _base(P):
    return cylinder(P, (0, BASE_Y, -BASE_H * 0.5), BASE_R, BASE_H * 0.5, rad=0.9)


# ------------------------------------------------------- costume decoration
def _ww_emblem(P, z0=107.6):
    """Stylised twin-W crest, drawn as a tapered polyline in the XZ plane."""
    p = proj(P, (0, 2), (0.0, z0))
    d = seg2(p, (-6.4, 3.1), (-2.6, -3.4), 0.55, 1.15)
    d = smin(d, seg2(p, (-2.6, -3.4), (0.0, 2.0), 1.15, 0.85), 0.45)
    d = smin(d, seg2(p, (0.0, 2.0), (2.6, -3.4), 0.85, 1.15), 0.45)
    d = smin(d, seg2(p, (2.6, -3.4), (6.4, 3.1), 1.15, 0.55), 0.45)
    return d


def _details(P, H, d, legs):
    """Costume decoration that has to hug the finished surface.

    P are world points, H the same points in the head frame.  Every decal is
    measured against `base`, a frozen copy of the body field: an embossed
    detail is only an approximate distance field, so feeding one into the next
    would let the error accumulate until details float off the surface."""
    base = d.copy()
    front = -(P[:, 1] - F(0.0))          # keep y > 0
    back = P[:, 1] - F(-1.0)             # keep y < -1
    z = P[:, 2]
    p = proj(P, (0, 2), (0.0, 0.0))

    raised = np.full_like(d, 1e3)
    # tiara: a full circlet with the star over the brow
    raised = np.minimum(raised, emboss(base, slab(H, 2, 142.2, 0.75), 0.6))
    su, sv = fuv(0.0, 142.3)
    raised = np.minimum(raised, emboss(
        base, star5(fp(H) - np.array([su, sv], F), 2.1), 0.9, -(H[:, 1] - F(3.0))))
    # chest crest
    raised = np.minimum(raised, emboss(base, _ww_emblem(P), 0.65, -(P[:, 1] - F(2.0))))
    # belt and its star
    raised = np.minimum(raised, emboss(base, slab(P, 2, 87.2, 1.8), 0.6))
    raised = np.minimum(raised, emboss(
        base, star5(proj(P, (0, 2), (0.0, 87.2)), 2.4), 0.95, front))
    # stars over the skirt
    for cx, cz, cl in ((0.0, 77.0, front), (-8.8, 79.5, front), (8.8, 79.5, front),
                       (-5.6, 78.0, back), (5.6, 78.0, back)):
        raised = np.minimum(raised, emboss(
            base, star5(proj(P, (0, 2), (cx, cz)), 1.6), 0.4, cl))
    # boot trim: a band along the rim plus a stripe down the front
    rim = np.maximum(np.abs(z - (_boot_top(P) - F(2.0))) - F(0.7), legs - F(1.9))
    raised = np.minimum(raised, emboss(base, rim, 0.4))
    for s in SIDES:
        raised = np.minimum(raised, emboss(
            base, seg2(p, (s * 10.9, 7.0), (s * 10.4, 34.0), 0.6), 0.4, front))
    d = np.minimum(d, raised)

    cut = np.full_like(d, 1e3)
    # leotard seams
    neck = smin(seg2(p, (-8.8, 115.0), (0.0, 109.8), 0.42),
                seg2(p, (0.0, 109.8), (8.8, 115.0), 0.42), 0.4)
    cut = np.minimum(cut, engrave(base, neck, 0.55, -(P[:, 1] - F(2.0))))
    for s in SIDES:
        legf = smin(seg2(p, (s * 14.4, 82.0), (s * 10.0, 74.0), 0.42),
                    seg2(p, (s * 10.0, 74.0), (s * 4.4, 68.0), 0.42), 0.5)
        cut = np.minimum(cut, engrave(base, legf, 0.55, front))
        cut = np.minimum(cut, engrave(
            base, seg2(p, (s * 13.4, 79.5), (s * 4.6, 71.0), 0.42), 0.55, back))
        for k in (-1.4, 1.4):                                  # bracer flutes
            a = (s * 20.7 + k * 0.4, -4.0 + k * 1.5, 92.6)
            b = (s * 16.0 + k * 0.4, 1.3 + k * 1.5, 87.3)
            cut = np.minimum(cut, engrave(base, capsule(P, a, b, 0.42), 0.5))
    d = smax(d, -cut, 0.3)

    # groove around the rim of the base
    ring = np.maximum(np.abs(np.hypot(P[:, 0], P[:, 1] - F(BASE_Y)) - F(22.6)) - F(0.55),
                      np.abs(z + F(0.6)) - F(2.5))
    return smax(d, -ring, 0.25)


# --------------------------------------------------------------------- API
def sdf(P):
    """Signed distance of the whole figure; P is (N, 3), result is (N,)."""
    P = np.ascontiguousarray(P, dtype=F)
    H = head_frame(P)

    legs = _legs(P)
    d = _torso(P)
    d = smin(d, legs, 3.2)
    d = smin(d, _arms(P), 2.6)
    d = smin(d, _head(H), 2.2)
    d = _face_cuts(H, d)
    d = _hand_cuts(P, d)
    d = np.minimum(d, _bracers(P))
    d = np.minimum(d, _boots(P, legs))
    d = smin(d, _hair(H), 1.6)
    d = _hair_strands(H, d)
    d = smin(d, _cape(P), 1.5)
    d = np.minimum(d, _lasso(P))
    d = _details(P, H, d, legs)
    d = smin(d, _base(P), 1.8)
    return d
