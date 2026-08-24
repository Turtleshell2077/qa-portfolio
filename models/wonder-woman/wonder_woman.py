"""Wonder Woman inspired collectible figure - procedural SDF sculpt.

Original geometry, built entirely from analytic signed distance fields.
The figure is caught mid-stride: right leg planted forward, left leg pushing
off, shoulders counter-rotated against the hips, hair and cape swept back.

All coordinates are millimetres.  The soles stand on z = 0, the display base
sits below that, the character walks towards +Y and +Z is up.  +X is her
right hand side.
"""
import numpy as np

from sdf_lib import (F, ln, smin, smax, rotmat, local, sphere, ellipsoid,
                     capsule, round_cone, box, torus, cylinder, chain, slab,
                     seg2, star5, circle2, proj, cyl_proj, cyl_uv, emboss,
                     engrave)
import head_lib


def aim(v):
    """Rotation whose local +Y axis points along the unit vector v."""
    return rotmat(rz=np.degrees(np.arctan2(-v[0], v[1])),
                  rx=np.degrees(np.arcsin(np.clip(float(v[2]), -1.0, 1.0))))

SIDES = (-1.0, 1.0)                      # -1 = her left, +1 = her right

BASE_R = 27.0
BASE_H = 4.0
BASE_Y = -3.0
BOUNDS = ((-34.0, 34.0), (-36.0, 30.0), (-BASE_H - 3.0, 153.0))

# --------------------------------------------------------------- skeleton
HIP = {1: (7.2, 2.4, 76.0), -1: (-7.0, -1.8, 76.0)}
KNEE = {1: (8.6, 11.0, 40.0), -1: (-10.0, -9.5, 39.0)}
ANKLE = {1: (7.8, 14.5, 9.0), -1: (-11.4, -16.0, 11.0)}
TOE = {1: (7.6, 21.0, 2.6), -1: (-11.2, -9.6, 2.2)}

SHOULDER = {1: (13.4, 1.2, 114.5), -1: (-14.2, 3.4, 115.3)}
ELBOW = {1: (16.8, -5.6, 97.0), -1: (-17.2, 1.2, 97.5)}
WRIST = {1: (15.2, -10.4, 82.5), -1: (-13.6, 8.4, 84.0)}

HEAD_O = np.array([1.2, 3.8, 134.6], dtype=F)
HEAD_R = rotmat(rx=3.0, rz=-7.0)

TORSO_AXIS_Y = 1.4                       # decal projection for the costume
TORSO_R = 12.0


def head_frame(P):
    return (P - HEAD_O) @ HEAD_R


def tp(P):
    return cyl_proj(P, TORSO_AXIS_Y, TORSO_R)


def tuv(x, z):
    return cyl_uv(x, z, TORSO_R)


# ------------------------------------------------------------------ torso
def _torso(P):
    d = ellipsoid(P, (0.3, 0.2, 78.0), (13.6, 8.9, 10.6), rotmat(rz=7))
    d = smin(d, ellipsoid(P, (0.2, -5.6, 79.5), (10.4, 5.2, 7.0), rotmat(rz=7)), 4.0)
    d = smin(d, ellipsoid(P, (0.5, 1.0, 92.0), (9.1, 7.0, 8.2), rotmat(rz=2)), 4.5)
    d = smin(d, ellipsoid(P, (0.6, 1.8, 105.0), (12.2, 8.2, 11.4), rotmat(rz=-5)), 4.5)
    d = smin(d, ellipsoid(P, (0.7, 1.6, 113.6), (11.3, 7.0, 5.5), rotmat(rz=-6)), 4.0)
    for s in SIDES:                                                    # bust
        d = smin(d, sphere(P, (s * 5.0 + 0.6, 7.0 - s * 0.5, 106.2), 4.8), 3.2)
    for s in SIDES:                                                    # deltoids
        d = smin(d, ellipsoid(P, SHOULDER[s], (5.0, 5.2, 5.6)), 4.2)
    d = smin(d, round_cone(P, (1.1, 3.6, 127.0), (0.6, 1.4, 115.5), 3.4, 6.2), 3.4)
    return d


def _abs_lines(P, d):
    """A hint of the abdominal division - the costume covers most of it."""
    q = tp(P)
    base = d.copy()
    cut = engrave(base, seg2(q, tuv(0.0, 98.0), tuv(0.0, 88.0), 0.34), 0.4,
                  -(P[:, 1] - F(3.0)))
    return smax(d, -cut, 0.35)


# ------------------------------------------------------------------- arms
def _arms(P):
    d = None
    for s in SIDES:
        S, E, W = SHOULDER[s], ELBOW[s], WRIST[s]
        a = round_cone(P, S, E, 4.4, 3.1)
        a = smin(a, round_cone(P, E, W, 3.3, 2.5), 2.2)
        # hand: a flattened wedge with a thumb rolled onto it
        v = np.array(W, F) - np.array(E, F)
        v /= np.linalg.norm(v)
        side = np.cross(v, np.array([0, 0, 1.0], F))
        side /= np.linalg.norm(side)
        Rh = aim(v)
        a = smin(a, box(P, tuple(np.array(W, F) + v * 3.1),
                        (1.15, 2.2, 1.5), 1.0, Rh), 1.6)
        a = smin(a, capsule(P, tuple(np.array(W, F) + v * 1.2 + side * s * 1.5),
                            tuple(np.array(W, F) + v * 3.4 + side * s * 1.1),
                            0.95), 1.1)
        d = a if d is None else np.minimum(d, a)
    return d


# ------------------------------------------------------------------- legs
def _legs(P):
    d = None
    for s in SIDES:
        H, K, A, T = HIP[s], KNEE[s], ANKLE[s], TOE[s]
        l = round_cone(P, H, K, 6.9, 4.5)
        l = smin(l, round_cone(P, K, A, 4.4, 2.8), 2.6)
        mid = tuple((np.array(K, F) * 0.62 + np.array(A, F) * 0.38)
                    + np.array([s * 0.6, -2.6, 0], F))
        l = smin(l, ellipsoid(P, mid, (3.7, 4.0, 7.2)), 3.2)             # calf
        l = smin(l, ellipsoid(P, tuple(np.array(K, F) + np.array([0, 1.0, 3.0], F)),
                              (4.3, 4.5, 4.6)), 3.0)                     # knee
        # foot: a sole slab aligned with the ankle-to-toe axis, plus a heel
        A3, T3 = np.array(A, F), np.array(T, F)
        v = T3 - A3
        L = float(np.linalg.norm(v))
        v = v / L
        Rf = aim(v)
        l = smin(l, box(P, tuple((A3 + T3) * 0.5), (1.5, L * 0.5 - 0.6, 0.7), 1.7, Rf), 1.6)
        l = smin(l, sphere(P, tuple(A3 - v * 1.2 + np.array([0, 0, -1.6], F)), 2.5), 2.0)
        d = l if d is None else np.minimum(d, l)
    return d


def _boot_top(P, s):
    """Rim of the over-the-knee boot, peaked at the front."""
    t = np.clip((P[:, 1] - F(KNEE[s][1])) / F(6.0), -1.0, 1.0)
    return F(KNEE[s][2] + 7.0) + F(4.4) * t


def _boots(P, legs):
    d = np.full_like(legs, 1e3)
    for s in SIDES:
        b = np.maximum(legs - F(1.6), P[:, 2] - _boot_top(P, s))
        b = np.maximum(b, F(-s) * P[:, 0])          # each boot on its own side
        d = np.minimum(d, b)
    return smax(d, -(P[:, 2] + F(BASE_H + 4.0)), 0.4)


# ------------------------------------------------------------------- head
def _head(P):
    return head_lib.head(head_frame(P))


# ------------------------------------------------------------------- hair
def _hair(P):
    """Long hair caught by the wind: swept back and to her left."""
    Q = head_frame(P)
    cap = ellipsoid(Q, (0, -0.6, 1.6), (7.5, 8.4, 9.0))
    cap = smin(cap, ellipsoid(Q, (0, -3.4, 2.4), (6.9, 6.6, 7.8)), 3.4)
    face = box(Q, (0, 11.0, -3.4), (3.9, 5.0, 5.6), 2.6)
    cap = smax(cap, -face, 1.2)
    d = cap
    # the mass gathered behind the head
    d = smin(d, ellipsoid(Q, (-1.2, -6.0, -3.0), (7.4, 6.2, 8.2)), 3.4)
    # streams of hair, each a tapered chain sweeping back and to her left
    locks = [
        [(5.6, -1.0, 5.0), (7.2, -4.0, 0.0), (7.0, -7.5, -6.0), (4.6, -11.0, -13.0),
         (0.5, -13.5, -19.0), (-5.0, -14.0, -23.5)],
        [(6.4, 0.6, 1.6), (7.6, -3.0, -3.4), (6.4, -7.0, -10.0), (2.6, -10.5, -16.5),
         (-3.0, -12.5, -21.0)],
        [(-5.8, -0.6, 5.2), (-7.6, -3.6, 0.4), (-8.6, -7.0, -5.4), (-8.4, -10.0, -12.0),
         (-7.0, -12.0, -18.5), (-4.6, -12.5, -24.0)],
        [(-6.6, 0.8, 1.4), (-8.4, -2.6, -3.6), (-9.4, -6.0, -10.4), (-9.0, -9.0, -17.0),
         (-7.6, -10.5, -23.0)],
        [(0.0, -5.0, 8.0), (1.6, -8.4, 3.0), (1.0, -11.0, -4.0), (-1.6, -12.6, -11.0),
         (-5.0, -13.0, -17.5), (-8.6, -12.0, -22.0)],
        [(3.0, -4.2, 7.0), (4.6, -8.0, 1.6), (3.6, -11.4, -5.6), (0.4, -13.4, -12.6),
         (-4.0, -14.0, -18.0)],
        [(-3.0, -4.4, 7.2), (-4.6, -8.2, 2.0), (-5.6, -11.4, -5.0), (-6.6, -13.0, -12.0),
         (-7.4, -13.0, -18.0)],
    ]
    radii = [(3.0, 3.2, 3.0, 2.6, 2.0, 1.1),
             (2.8, 3.0, 2.6, 2.0, 1.0),
             (3.0, 3.2, 3.0, 2.6, 2.0, 1.1),
             (2.8, 3.0, 2.6, 2.0, 1.0),
             (3.2, 3.4, 3.2, 2.8, 2.2, 1.2),
             (2.8, 3.0, 2.8, 2.2, 1.1),
             (2.8, 3.0, 2.8, 2.2, 1.1)]
    for pts, rad in zip(locks, radii):
        d = smin(d, chain(Q, pts, rad, k=2.6), 3.2)
    return d


def _hair_strands(P, d):
    Q = head_frame(P)
    base = d.copy()
    cut = np.full_like(d, 1e3)
    hair_only = np.minimum(Q[:, 1] - F(3.0), F(6.4) - Q[:, 2])
    for a in (-42.0, -28.0, -14.0, 0.0, 14.0, 28.0, 42.0):
        g = torus(Q, (0, -0.6, 1.8), 8.7, 0.4, rotmat(ry=90, rz=a))
        cut = np.minimum(cut, engrave(base, np.maximum(
            np.maximum(g, F(-7.0) - Q[:, 2]), hair_only), 0.55))
    # grooves running along the streaming locks
    for x0, y0, z0, x1, y1, z1 in (
            (6.4, -2.0, 3.0, -2.0, -13.0, -20.0),
            (4.0, -4.0, 5.0, -5.5, -13.5, -18.0),
            (-6.6, -2.0, 3.0, -6.0, -12.5, -21.0),
            (-4.0, -5.0, 5.0, -8.0, -12.0, -19.0),
            (1.0, -6.0, 6.0, -7.0, -12.5, -20.0)):
        cut = np.minimum(cut, engrave(
            base, capsule(Q, (x0, y0, z0), (x1, y1, z1), 0.42), 0.6))
    return smax(d, -cut, 0.45)


# ------------------------------------------------------------------- cape
def _cape(P):
    """A short mantle streaming back off the shoulders.  It deliberately stops
    above the knees: a full length cape hides the stride completely."""
    x = P[:, 0]
    y = P[:, 1] - F(2.0)
    z = P[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(x, y)
    t = np.clip((F(114.0) - z) / F(60.0), 0.0, 1.0)

    R = F(12.0) + F(14.5) * t ** 1.3
    R = (R + F(2.2) * np.sin(F(8.0) * th + F(0.4)) * t
         + F(1.0) * np.sin(F(13.0) * th - F(1.0)) * t * t
         + F(3.4) * np.sin(th) * t)               # blown towards her left
    lim = np.radians(F(108.0) + F(30.0) * np.clip((z - F(70.0)) / F(40.0), 0.0, 1.0))
    edge = np.clip((np.abs(th) - lim) / F(0.45), 0.0, 1.0)
    tuck = np.clip((z - F(92.0)) / F(14.0), 0.0, 1.0)
    top = np.clip((F(112.0) - z) / F(9.0), 0.0, 1.0)
    R = R * (F(1.0) - F(0.4) * (F(1.0) - edge) * tuck) * (F(0.55) + F(0.45) * top)

    half = (F(1.9) - F(0.45) * t) * F(0.5)
    d = np.abs(r - R) - half
    d = smax(d, -(np.abs(th) - lim) * np.minimum(r, F(14.0)), 0.9)
    lift = (1.0 - np.clip((np.abs(th) - lim) / F(0.42), 0.0, 1.0)) ** 2
    hem = F(54.0) + F(40.0) * lift - F(4.5) * np.sin(F(8.0) * th + F(0.4))
    d = smax(d, hem - z, 1.2)
    d = smax(d, z - F(116.0), 0.8)

    coll = torus(P, (0.6, 0.4, 118.0), 6.4, 1.5, rotmat(rx=12, rz=-6))
    coll = np.maximum(coll, P[:, 1] - F(1.6))
    return np.minimum(d, coll)


# ------------------------------------------------------------------ lasso
def _lasso(P):
    R = rotmat(rx=90, rz=-14, ry=10)
    c = (14.6, 5.6, 74.0)
    d = torus(P, c, 4.8, 0.9, R)
    d = np.minimum(d, torus(P, (c[0] + 0.4, c[1] - 1.4, c[2] + 0.4), 4.0, 0.78, R))
    d = np.minimum(d, capsule(P, (13.8, 4.8, 78.6), (13.0, 3.4, 84.0), 0.72))
    return d


# ------------------------------------------------------------------- base
def _base(P):
    return cylinder(P, (0, BASE_Y, -BASE_H * 0.5), BASE_R, BASE_H * 0.5, rad=0.9)


# ------------------------------------------------------------------ armour
def _crest(P, z0=107.0):
    """Stylised twin-W crest."""
    p = proj(P, (0, 2), (0.6, z0))
    d = seg2(p, (-6.4, 3.1), (-2.6, -3.4), 0.55, 1.15)
    d = smin(d, seg2(p, (-2.6, -3.4), (0.0, 2.0), 1.15, 0.85), 0.45)
    d = smin(d, seg2(p, (0.0, 2.0), (2.6, -3.4), 0.85, 1.15), 0.45)
    d = smin(d, seg2(p, (2.6, -3.4), (6.4, 3.1), 1.15, 0.55), 0.45)
    return d


def _bustier_top(u):
    """Sweetheart neckline: dips at the sternum, rises over each breast and
    falls away across the back."""
    back = np.clip((np.abs(u) - F(15.0)) / F(9.0), 0.0, 1.0)
    return (F(108.8) + F(3.4) * np.exp(-((u - F(5.6)) / F(4.2)) ** 2)
            + F(3.4) * np.exp(-((u + F(5.6)) / F(4.2)) ** 2)
            - F(6.5) * back)


def _armour(P, d, legs, arms):
    """Plate armour laid over the body as surface relief."""
    base = d.copy()
    q = tp(P)
    u, v = q[:, 0], q[:, 1]
    front = -(P[:, 1] - F(1.0))
    torso_only = F(1.4) - arms          # the wrap must not creep onto the arms
    z = P[:, 2]

    raised = np.full_like(d, 1e3)
    # --- bustier -------------------------------------------------------
    top = _bustier_top(u)
    bust = np.maximum(v - top, F(88.6) - v)
    raised = np.minimum(raised, emboss(base, bust, 0.5, torso_only))
    raised = np.minimum(raised, emboss(
        base, np.abs(v - top) - F(0.65), 0.75, torso_only))
    raised = np.minimum(raised, emboss(base, _crest(P), 0.75, -(P[:, 1] - F(3.0))))
    # --- belt ----------------------------------------------------------
    raised = np.minimum(raised, emboss(
        base, np.abs(v - F(87.4)) - F(1.9), 0.75, torso_only))
    raised = np.minimum(raised, emboss(
        base, star5(proj(P, (0, 2), (0.6, 87.4)), 2.4), 1.0, front))
    # --- skirt plates --------------------------------------------------
    skirt = np.maximum(v - F(85.4), F(70.5) + F(2.2) * np.cos(u / F(5.0)) - v)
    raised = np.minimum(raised, emboss(base, skirt, 0.55, torso_only))
    # --- bracers -------------------------------------------------------
    for s in SIDES:
        E, W = np.array(ELBOW[s], F), np.array(WRIST[s], F)
        a = tuple(E + (W - E) * 0.28)
        b = tuple(E + (W - E) * 0.92)
        raised = np.minimum(raised, round_cone(P, a, b, 4.3, 3.6))
    # --- boots: knee plate, rim and shin ribs --------------------------
    for s in SIDES:
        K = np.array(KNEE[s], F)
        raised = np.minimum(raised, emboss(base, np.maximum(
            np.abs(z - (_boot_top(P, s) - F(2.1))) - F(0.8),
            np.maximum(legs - F(2.4), F(-s) * P[:, 0])), 0.5))
        raised = np.minimum(raised, ellipsoid(
            P, tuple(K + np.array([0, 2.0, 3.4], F)), (3.6, 3.4, 3.6)))
        A3, K3 = np.array(ANKLE[s], F), np.array(KNEE[s], F)
        fwd = np.array([0.0, 1.0, 0.0], F)
        for k in (-1.0, 0.0, 1.0):
            off = np.array([k * 1.9, 0.0, 0.0], F)
            raised = np.minimum(raised, np.maximum(
                capsule(P, tuple(A3 + off + fwd * 1.4),
                        tuple(K3 * 0.72 + A3 * 0.28 + off + fwd * 1.4), 0.7),
                F(-s) * P[:, 0]))
        # cuff where the boot meets the calf
        raised = np.minimum(raised, emboss(base, np.maximum(
            np.abs(z - (_boot_top(P, s) - F(6.0))) - F(0.9),
            np.maximum(legs - F(2.6), F(-s) * P[:, 0])), 0.4))
    d = np.minimum(d, raised)

    cut = np.full_like(d, 1e3)
    # radiating panel lines over the bustier
    for a in (-3, -2, -1, 1, 2, 3):
        cut = np.minimum(cut, engrave(base, seg2(
            q, (0.6, 99.0), (0.6 + a * 6.2, 91.0 + abs(a) * 1.2), 0.28), 0.45))
    cut = np.minimum(cut, engrave(base, seg2(q, (0.6, 100.5), (0.6, 89.5), 0.3), 0.45))
    # separations between the skirt plates
    for i in range(-7, 8):
        uu = F(i) * F(5.0) + F(0.6)
        cut = np.minimum(cut, engrave(base, seg2(
            q, (uu, 85.0), (uu * 1.12, 71.0), 0.3), 0.5))
    # bracer flutes
    for s in SIDES:
        E, W = np.array(ELBOW[s], F), np.array(WRIST[s], F)
        n = np.cross(W - E, np.array([0, 0, 1.0], F))
        n /= np.linalg.norm(n)
        for k in (-1.0, 0.0, 1.0):
            off = n * (k * 2.4) + np.array([0, 0, k * 1.2], F)
            cut = np.minimum(cut, engrave(base, capsule(
                P, tuple(E + (W - E) * 0.33 + off), tuple(E + (W - E) * 0.88 + off),
                0.4), 0.5))
    d = smax(d, -cut, 0.3)

    ring = np.maximum(np.abs(np.hypot(P[:, 0], P[:, 1] - F(BASE_Y)) - F(23.4)) - F(0.55),
                      np.abs(z + F(0.6)) - F(2.5))
    return smax(d, -ring, 0.25)


def _tiara(P, d):
    Q = head_frame(P)
    base = d.copy()
    band = slab(Q, 2, 5.4, 0.75)
    r = emboss(base, band, 0.55)
    su, sv = head_lib.duv(0.0, 5.5)
    r = np.minimum(r, emboss(base, star5(
        head_lib.dp(Q) - np.array([su, sv], F), 2.1), 0.85, -(Q[:, 1] - F(3.0))))
    return np.minimum(d, r)


# --------------------------------------------------------------------- API
def sdf(P):
    P = np.ascontiguousarray(P, dtype=F)
    legs = _legs(P)
    arms = _arms(P)
    d = _torso(P)
    d = smin(d, legs, 3.2)
    d = smin(d, arms, 2.6)
    d = smin(d, _head(P), 2.4)
    d = _abs_lines(P, d)
    d = np.minimum(d, _boots(P, legs))
    d = smin(d, _hair(P), 1.6)
    d = _hair_strands(P, d)
    d = smin(d, _cape(P), 1.5)
    d = np.minimum(d, _lasso(P))
    d = _armour(P, d, legs, arms)
    d = _tiara(P, d)
    d = smin(d, _base(P), 1.8)
    return d
