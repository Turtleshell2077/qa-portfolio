"""Polygonise the SDF sculpt into a watertight, print-ready STL."""
import argparse
import time

import numpy as np
import trimesh
from skimage import measure

import wonder_woman as ww

F = np.float32


def sample_volume(vox, bounds=None, chunk=24, verbose=True):
    bounds = bounds or ww.BOUNDS
    axes = [np.arange(lo, hi + vox * 0.5, vox, dtype=F) for lo, hi in bounds]
    nx, ny, nz = (len(a) for a in axes)
    vol = np.empty((nx, ny, nz), dtype=F)
    X, Y = np.meshgrid(axes[0], axes[1], indexing='ij')
    flatXY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    t0 = time.time()
    for k0 in range(0, nz, chunk):
        k1 = min(k0 + chunk, nz)
        zs = axes[2][k0:k1]
        P = np.empty((flatXY.shape[0] * len(zs), 3), dtype=F)
        P[:, :2] = np.repeat(flatXY, len(zs), axis=0)
        P[:, 2] = np.tile(zs, flatXY.shape[0])
        vol[:, :, k0:k1] = ww.sdf(P).reshape(nx, ny, len(zs))
        if verbose:
            print(f'  slab {k1:>4}/{nz}  {time.time() - t0:6.1f}s', end='\r')
    if verbose:
        print(f'  sampled {nx}x{ny}x{nz} = {vol.size / 1e6:.1f}M voxels '
              f'in {time.time() - t0:.1f}s')
    return vol, axes


def check_sealed(vol):
    """The surface must not touch the edge of the grid or the mesh gets holes."""
    faces = [vol[0], vol[-1], vol[:, 0], vol[:, -1], vol[:, :, 0], vol[:, :, -1]]
    worst = min(float(f.min()) for f in faces)
    return worst


def polygonise(vol, axes, vox):
    verts, faces, normals, _ = measure.marching_cubes(
        vol, level=0.0, spacing=(vox, vox, vox), allow_degenerate=False)
    verts += np.array([a[0] for a in axes], dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def finish(mesh, target_height, target_faces=None):
    if target_faces and len(mesh.faces) > target_faces:
        try:
            small = mesh.simplify_quadric_decimation(face_count=target_faces)
        except TypeError:
            small = mesh.simplify_quadric_decimation(target_faces)
        small.merge_vertices()
        if not small.is_watertight:
            trimesh.repair.fill_holes(small)
        small.fix_normals()
        # never trade a sealed mesh for a lighter one
        if small.is_watertight and small.is_winding_consistent:
            mesh = small
        else:
            print('  decimation broke the seal - keeping the full resolution mesh')
    ext = mesh.bounds[1] - mesh.bounds[0]
    mesh.apply_scale(target_height / ext[2])
    mesh.apply_translation([-mesh.bounds[0][0] - (mesh.bounds[1][0] - mesh.bounds[0][0]) / 2,
                            -mesh.bounds[0][1] - (mesh.bounds[1][1] - mesh.bounds[0][1]) / 2,
                            -mesh.bounds[0][2]])
    return mesh


def report(mesh):
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f'  triangles : {len(mesh.faces):,}')
    print(f'  vertices  : {len(mesh.vertices):,}')
    print(f'  size      : {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm')
    print(f'  watertight: {mesh.is_watertight}   winding ok: {mesh.is_winding_consistent}')
    print(f'  volume    : {mesh.volume / 1000.0:.1f} cm3   euler: {mesh.euler_number}')
    print(f'  bodies    : {mesh.body_count}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voxel', type=float, default=0.35)
    ap.add_argument('--height', type=float, default=150.0)
    ap.add_argument('--faces', type=int, default=0, help='decimate to N faces')
    ap.add_argument('--out', default='wonder_woman.stl')
    ap.add_argument('--crop', default=None,
                    help='z0,z1 - build only that slice (inspection only)')
    ap.add_argument('--min-part', type=float, default=2.0,
                    help='drop disconnected shards below this volume in mm3')
    args = ap.parse_args()

    bounds = ww.BOUNDS
    if args.crop:
        z0, z1 = (float(v) for v in args.crop.split(','))
        bounds = (bounds[0], bounds[1], (z0, z1))
    print(f'sampling at {args.voxel} mm ...')
    vol, axes = sample_volume(args.voxel, bounds)
    if not args.crop:
        print(f'  clearance to grid wall: {check_sealed(vol):.2f} mm')
    print('marching cubes ...')
    mesh = polygonise(vol, axes, args.voxel)
    del vol
    if args.min_part > 0 and not args.crop:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            keep = [p for p in parts if abs(p.volume) >= args.min_part]
            dropped = len(parts) - len(keep)
            mesh = trimesh.util.concatenate(keep)
            mesh.merge_vertices()
            print(f'  dropped {dropped} shard(s) under {args.min_part} mm3')
    if not args.crop:
        mesh = finish(mesh, args.height, args.faces or None)
    report(mesh)
    mesh.export(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
