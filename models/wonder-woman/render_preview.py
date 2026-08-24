"""Quick clay-shaded turntable renders of a mesh (numpy point splatting)."""
import argparse

import numpy as np
import trimesh
from PIL import Image


def sample_surface(mesh, n, rng):
    areas = mesh.area_faces
    idx = rng.choice(len(areas), size=n, p=areas / areas.sum())
    tri = mesh.triangles[idx]
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v) > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    pts = tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])
    return pts, mesh.face_normals[idx]


def view_matrix(az, el):
    a, e = np.radians(az), np.radians(el)
    cd = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    fwd = -cd
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.stack([right, up, cd])


def render(mesh, az=0.0, el=8.0, W=460, H=740, samples=6_000_000, chunk=750_000,
           focus=None, span=None):
    M = view_matrix(az, el)
    c = mesh.bounds.mean(axis=0)
    ext = mesh.bounds[1] - mesh.bounds[0]
    if focus is not None:
        c = c.copy()
        c[2] = focus
    scale = (H * 0.92) / (span if span else ext[2])
    depth = np.full(W * H, np.inf)
    nbuf = np.zeros((W * H, 3))
    rng = np.random.default_rng(7)
    for start in range(0, samples, chunk):
        n = min(chunk, samples - start)
        pts, nrm = sample_surface(mesh, n, rng)
        q = (pts - c) @ M.T
        px = np.rint(q[:, 0] * scale + W / 2).astype(np.int64)
        py = np.rint(H / 2 - q[:, 1] * scale).astype(np.int64)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        pix = py[ok] * W + px[ok]
        dz = -q[ok, 2]        # distance from the camera, near side wins
        np.minimum.at(depth, pix, dz)
        keep = dz <= depth[pix] + 1e-9
        nbuf[pix[keep]] = (nrm[ok][keep]) @ M.T
    hit = np.isfinite(depth)
    n = nbuf.copy()
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, np.maximum(ln, 1e-9))
    key = np.array([-0.42, 0.55, 0.72])
    key /= np.linalg.norm(key)
    fill = np.array([0.55, 0.25, 0.20])
    fill /= np.linalg.norm(fill)
    lam = np.clip(n @ key, 0, 1) * 0.82 + np.clip(n @ fill, 0, 1) * 0.24
    rim = np.clip(1.0 - np.abs(n[:, 2]), 0, 1) ** 3 * 0.35
    shade = np.clip(0.13 + lam + rim, 0, 1) ** (1 / 2.2)
    img = np.where(hit[:, None], shade[:, None] * np.array([0.93, 0.90, 0.88]),
                   np.array([0.10, 0.11, 0.13]))
    return Image.fromarray((img.reshape(H, W, 3) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh')
    ap.add_argument('--out', default='preview.png')
    ap.add_argument('--views', default='0,35,90,180')
    ap.add_argument('--samples', type=int, default=6_000_000)
    ap.add_argument('--focus', type=float, default=None)
    ap.add_argument('--span', type=float, default=None)
    ap.add_argument('--size', default='460x740')
    args = ap.parse_args()
    mesh = trimesh.load(args.mesh)
    W, H = (int(v) for v in args.size.split('x'))
    tiles = [render(mesh, az=float(a), samples=args.samples, W=W, H=H,
                    focus=args.focus, span=args.span)
             for a in args.views.split(',')]
    W, H = tiles[0].size
    sheet = Image.new('RGB', (W * len(tiles), H))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * W, 0))
    sheet.save(args.out)
    print('wrote', args.out, sheet.size)


if __name__ == '__main__':
    main()


