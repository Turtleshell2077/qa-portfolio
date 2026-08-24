"""The head, built in its own frame.

Local coordinates: origin at the centre of the skull, +Y is the way she looks,
+Z is up, 1 unit = 1 mm.  The canonical landmarks follow the classical
proportions of a head 19 mm tall (crown +10.2, chin -8.8):

    hairline  +6.9      brow      +1.7      eye line  +0.7
    nose base -3.6      mouth     -5.3      chin      -8.8
"""
import numpy as np

from sdf_lib import (F, smin, smax, rotmat, local, sphere, ellipsoid, capsule,
                     round_cone, box, torus, chain, shell, lens, seg2,
                     cyl_proj, cyl_uv, emboss, engrave)

SIDES = (-1.0, 1.0)

CROWN = 10.2
CHIN = -8.8
SKULL = (6.9, 8.0, 9.0)         # radii of the cranial mass
SKULL_C = (0.0, -0.8, 1.2)

EYE = (3.00, 5.30, 0.7)         # eyeball centre (x mirrored)
EYE_BALL = 1.38
EYE_LID = 2.00

DECAL_AXIS_Y = 0.4              # decal projection: vertical axis and radius
DECAL_R = 6.9


def dp(Q):
    return cyl_proj(Q, DECAL_AXIS_Y, DECAL_R)


def duv(x, z):
    return cyl_uv(x, z, DECAL_R)


# ----------------------------------------------------------------- masses
def skull(Q):
    """Cranium plus a separate facial mass.  Without the second volume the
    front of the head falls away below the eyes and every feature ends up
    sitting on it as a lump."""
    d = ellipsoid(Q, SKULL_C, SKULL)                                     # cranium
    d = smin(d, ellipsoid(Q, (0, -3.0, 1.6), (6.5, 7.2, 7.8)), 3.2)      # occiput
    # a rounded slab, not an ellipsoid: it holds the facial plane flat from
    # the brow down to the chin instead of falling away below the eyes
    d = smin(d, box(Q, (0, 3.2, -3.4), (1.6, 1.0, 2.4), 2.8), 2.6)       # face mass
    d = smin(d, ellipsoid(Q, (0, 1.4, 4.8), (5.2, 6.0, 3.4)), 2.6)       # forehead
    # taper the whole lower half towards the chin
    d = smax(d, round_cone(Q, (0, 1.0, 5.0), (0, 3.6, -9.2), 8.7, 2.9), 2.2)
    for s in SIDES:
        d = smin(d, ellipsoid(Q, (s * 4.2, 4.2, 0.4), (2.5, 2.3, 1.7)), 2.2)  # zygomatic
        d = smin(d, ellipsoid(Q, (s * 3.6, 2.6, -3.9), (2.2, 2.6, 2.8)), 2.4)  # cheek
        d = smin(d, capsule(Q, (s * 1.3, 5.5, -7.8), (s * 3.6, 1.2, -5.0), 0.85), 2.4)
        d = smin(d, round_cone(Q, (s * 0.8, 7.1, 2.2),
                               (s * 4.3, 5.9, 2.4), 0.4, 0.2), 1.6)      # brow ridge
        d = smin(d, ellipsoid(Q, (s * 6.4, 0.0, -1.1), (0.8, 1.5, 2.0)), 1.0)  # ear
    d = smin(d, ellipsoid(Q, (0, 5.7, -7.3), (1.7, 1.2, 1.3)), 2.0)      # chin
    # nose
    d = smin(d, round_cone(Q, (0, 6.5, 2.2), (0, 8.05, -2.35), 0.32, 0.5), 1.0)
    d = smin(d, sphere(Q, (0, 8.3, -3.1), 0.6), 0.8)                     # tip
    d = smin(d, ellipsoid(Q, (0, 7.75, -3.55), (1.45, 0.62, 0.5)), 0.8)  # wings
    d = smin(d, capsule(Q, (0, 7.9, -3.6), (0, 7.55, -4.1), 0.25), 0.4)  # septum
    # lips laid on the muzzle
    d = smin(d, ellipsoid(Q, (0, 7.05, -4.66), (1.8, 0.45, 0.32)), 0.85)  # upper
    d = smin(d, ellipsoid(Q, (0, 7.1, -5.6), (1.55, 0.5, 0.42)), 0.85)    # lower
    return d


def eyes(Q, d):
    """Eyeball in a socket with lids wrapped around it - the lids are a
    spherical shell with an almond aperture cut out, which reads as an open
    eye far better than any added-on ridge."""
    for s in SIDES:
        c = (s * EYE[0], EYE[1], EYE[2])
        # the globe sits a little in front of the lid centre so it fills the
        # aperture instead of reading as an empty socket
        cb = (s * EYE[0], EYE[1] + 0.30, EYE[2])
        ac = (s * EYE[0], EYE[1], EYE[2] - 0.06)
        d = smax(d, -sphere(Q, c, EYE_LID), 0.2)                         # socket
        ball = sphere(Q, cb, 1.48)
        lids = shell(Q, c, EYE_LID, 1.40)
        # the aperture axis points slightly down, which is what makes the
        # upper lid overhang the globe and the lower lid fall away
        R = rotmat(rx=-11.0, rz=s * -9.0)
        lids = smax(lids, -lens(Q, ac, 0.80, 1.66, R), 0.14)             # aperture
        d = smin(d, np.minimum(ball, lids), 0.12)
    return d


def details(Q, d):
    """Brows, lid crease, mouth line, philtrum and nostrils."""
    base = d.copy()
    front = -(Q[:, 1] - F(3.0))
    q = dp(Q)
    raised = np.full_like(d, 1e3)
    for s in SIDES:
        raised = np.minimum(raised, emboss(base, smin(
            seg2(q, duv(s * 1.1, 1.95), duv(s * 3.0, 2.4), 0.17, 0.14),
            seg2(q, duv(s * 3.0, 2.4), duv(s * 4.6, 1.95), 0.14, 0.08), 0.18), 0.11, front))
    d = np.minimum(d, raised)

    cut = np.full_like(d, 1e3)
    for s in SIDES:
        cut = np.minimum(cut, engrave(base, smin(          # lid crease
            seg2(q, duv(s * 1.5, 1.5), duv(s * 3.1, 1.72), 0.13),
            seg2(q, duv(s * 3.1, 1.72), duv(s * 4.5, 1.3), 0.13), 0.22), 0.25, front))
    # mouth line with the corners turned down a touch
    mouth = smin(seg2(q, duv(-2.0, -5.3), duv(0.0, -5.05), 0.16),
                 seg2(q, duv(0.0, -5.05), duv(2.0, -5.3), 0.16), 0.25)
    cut = np.minimum(cut, engrave(base, mouth, 0.32, front))
    for s in SIDES:
        cut = np.minimum(cut, engrave(base, ellipsoid(                   # nostril
            Q, (s * 0.72, 7.95, -4.1), (0.24, 0.46, 0.2),
            rotmat(rx=-22)), 0.4))
        cut = np.minimum(cut, engrave(base, capsule(                     # alar crease
            Q, (s * 1.45, 7.6, -3.15), (s * 1.15, 7.5, -4.05), 0.14), 0.25))
        cut = np.minimum(cut, engrave(base, capsule(                     # ear bowl
            Q, (s * 6.8, 0.5, 0.3), (s * 6.8, 0.7, -1.6), 0.55), 0.6))
    return smax(d, -cut, 0.2)


def head(Q):
    d = skull(Q)
    d = eyes(Q, d)
    d = details(Q, d)
    return d
