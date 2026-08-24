"""Fast render of the head alone, straight from its local frame."""
import argparse
import numpy as np
import trimesh
from skimage import measure

import head_lib
import render_preview as rp

ap = argparse.ArgumentParser()
ap.add_argument('--voxel', type=float, default=0.15)
ap.add_argument('--out', default='/tmp/face.png')
ap.add_argument('--views', default='0,25,55,90')
ap.add_argument('--fn', default='head')
ap.add_argument('--size', default='340x440')
ap.add_argument('--samples', type=int, default=4_000_000)
ap.add_argument('--focus', type=float, default=None)
ap.add_argument('--span', type=float, default=None)
args = ap.parse_args()

F = np.float32
bounds = ((-11, 11), (-12, 13), (-17, 12))
axes = [np.arange(a, b + args.voxel * .5, args.voxel, dtype=F) for a, b in bounds]
nx, ny, nz = (len(a) for a in axes)
vol = np.empty((nx, ny, nz), F)
X, Y = np.meshgrid(axes[0], axes[1], indexing='ij')
flat = np.stack([X.ravel(), Y.ravel()], -1)
fn = getattr(head_lib, args.fn)
for k0 in range(0, nz, 32):
    zs = axes[2][k0:k0 + 32]
    P = np.empty((flat.shape[0] * len(zs), 3), F)
    P[:, :2] = np.repeat(flat, len(zs), axis=0)
    P[:, 2] = np.tile(zs, flat.shape[0])
    vol[:, :, k0:k0 + len(zs)] = fn(P).reshape(nx, ny, len(zs))
v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(args.voxel,) * 3)
v += np.array([a[0] for a in axes])
m = trimesh.Trimesh(v, f, process=True)
m.merge_vertices(); m.fix_normals()
print('tris', len(m.faces), 'watertight', m.is_watertight)
W, H = (int(v) for v in args.size.split('x'))
tiles = [rp.render(m, az=float(a), W=W, H=H, samples=args.samples,
                   focus=args.focus, span=args.span)
         for a in args.views.split(',')]
from PIL import Image
out = Image.new('RGB', (W * len(tiles), H))
for i, t in enumerate(tiles):
    out.paste(t, (i * W, 0))
out.save(args.out)
print('wrote', args.out)
