import argparse
import os

import cv2
import nibabel as nib
import numpy as np
import vedo
from scipy.ndimage import map_coordinates


class EmptySliceMaskError(ValueError):
    pass


def get_effective_affine(img, multiply_affine_by_zooms: bool):
    affine = img.affine.astype(np.float32, copy=True)
    if multiply_affine_by_zooms:
        zooms = img.header.get_zooms()[:3]
        for axis_idx, zoom in enumerate(zooms):
            column = affine[:3, axis_idx]
            norm = float(np.linalg.norm(column))
            if norm > 1e-6:
                affine[:3, axis_idx] = column / norm * float(zoom)
            else:
                affine[:3, axis_idx] = 0.0
                affine[axis_idx, axis_idx] = float(zoom)
    return affine


def long_axis_endpoints_from_nifti(
    nii_gz_path,
    label_value: int | None = None,
    max_points: int = 500_000,
    multiply_affine_by_zooms: bool = False,
):
    img = nib.load(nii_gz_path) if isinstance(nii_gz_path, str) else nii_gz_path
    seg = img.get_fdata()
    affine = get_effective_affine(img, multiply_affine_by_zooms=multiply_affine_by_zooms)

    mask = seg > 0 if label_value is None else seg == label_value
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        raise ValueError("Mask is empty.")

    if ijk.shape[0] > max_points:
        sel = np.random.choice(ijk.shape[0], max_points, replace=False)
        ijk = ijk[sel]

    xyz = (affine @ np.c_[ijk, np.ones((ijk.shape[0], 1))].T).T[:, :3]
    mean = xyz.mean(axis=0)
    axis = affine[:3, 2].astype(float)
    axis /= np.linalg.norm(axis)
    proj = (xyz - mean) @ axis

    p_start = mean + proj.min() * axis
    p_end = mean + proj.max() * axis
    return tuple(p_start), tuple(p_end)


def _orthonormal_basis_from_z(z: np.ndarray):
    z = z / (np.linalg.norm(z) + 1e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return x, y, z


def random_rotation_limited_axis(vec, max_angle_deg):
    v = np.asarray(vec, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12)
    max_angle = np.deg2rad(max_angle_deg)

    u = np.random.uniform(0.0, 1.0)
    theta = np.arccos(1.0 - u * (1.0 - np.cos(max_angle)))
    psi = np.random.uniform(0.0, 2.0 * np.pi)

    e1, e2, _ = _orthonormal_basis_from_z(v)
    w = np.cos(theta) * v + np.sin(theta) * (np.cos(psi) * e1 + np.sin(psi) * e2)
    w /= np.linalg.norm(w) + 1e-12

    phi = np.random.uniform(0.0, 2.0 * np.pi)
    x0, y0, z0 = _orthonormal_basis_from_z(w)
    x = np.cos(phi) * x0 + np.sin(phi) * y0
    y = -np.sin(phi) * x0 + np.cos(phi) * y0
    return np.stack([x, y, z0], axis=1)


def get_rotated_bbox(mask, direction):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise EmptySliceMaskError("Sampled slice does not contain any foreground mask pixels.")
    pts = np.vstack([xs, ys]).T

    u = np.array(direction, dtype=float)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0])
    else:
        u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])

    proj_u = pts @ u
    proj_v = pts @ v
    umin, umax = proj_u.min(), proj_u.max()
    vmin, vmax = proj_v.min(), proj_v.max()

    corners_uv = np.array([[umin, vmin], [umin, vmax], [umax, vmax], [umax, vmin]])
    corners_xy = corners_uv[:, 0:1] * u.reshape(1, 2) + corners_uv[:, 1:2] * v.reshape(1, 2)

    center_xy = ((umin + umax) / 2) * u + ((vmin + vmax) / 2) * v
    x, y = center_xy.tolist()
    w = umax - umin
    h = vmax - vmin
    t = np.degrees(np.arctan2(u[1], u[0]))
    return (x, y, w, h, t), corners_xy


def get_one_slice(
    nii,
    gt,
    point_ras,
    bas_ras,
    tip_ras,
    label_value: int,
    slice_size,
    max_angle_deg: float,
    multiply_affine_by_zooms: bool = False,
    visualize=False,
):
    rotation = random_rotation_limited_axis(tip_ras - bas_ras, max_angle_deg)
    bas_local = np.matmul(bas_ras - point_ras, rotation)
    tip_local = np.matmul(tip_ras - point_ras, rotation)

    resolution = np.random.uniform(1.0, 2.0)
    data = nii.get_fdata()
    data_gt = gt.get_fdata() == label_value
    affine = get_effective_affine(nii, multiply_affine_by_zooms=multiply_affine_by_zooms)
    inv_affine = np.linalg.inv(affine)

    u = rotation[:, 0]
    v = rotation[:, 1]

    h, w = slice_size
    uu, vv = np.meshgrid(
        np.linspace(-w // 2, w // 2, w) * resolution,
        np.linspace(-h // 2, h // 2, h) * resolution,
    )
    uu += np.random.uniform(-w // 4, w // 4)
    vv += np.random.uniform(-h // 4, h // 4)

    ras_points = point_ras.reshape(3, 1, 1) + u.reshape(3, 1, 1) * uu + v.reshape(3, 1, 1) * vv
    ras_homo = np.vstack([ras_points.reshape(3, -1), np.ones((1, ras_points.size // 3))])
    ijk = (inv_affine @ ras_homo)[:3, :].reshape(3, h, w)

    slice_data = map_coordinates(data, [ijk[0], ijk[1], ijk[2]], order=1, mode="nearest", cval=0)
    slice_data_gt = map_coordinates(data_gt, [ijk[0], ijk[1], ijk[2]], order=0, mode="constant", cval=0)

    direction = tip_local - bas_local
    direction /= np.linalg.norm(direction)
    if direction[2] < 0:
        direction[2] *= -1
    rbox, qbox = get_rotated_bbox(slice_data_gt, direction[:2])

    slice_norm = slice_data - slice_data.min()
    slice_norm /= slice_norm.max() + 1e-8
    slice_uint8 = (slice_norm * 255).astype(np.uint8)

    vis = None
    if visualize:
        img = vedo.Image(slice_uint8)
        box_points_3d = np.c_[qbox, np.zeros(len(qbox))]
        lines = [
            vedo.Line(box_points_3d[i], box_points_3d[(i + 1) % 4], c="yellow", lw=4)
            for i in range(4)
        ]
        p1 = np.array([rbox[0], rbox[1], 0])
        arrow = vedo.Arrow(p1, p1 + 50 * direction, s=0.5)

        plt = vedo.show(img, arrow, *lines, offscreen=True)
        plt.camera.OrthogonalizeViewUp()
        vis = plt.screenshot(asarray=True)
        vis = vis[
            vis.shape[0] // 2 + 314:vis.shape[0] // 2 - 314:-1,
            vis.shape[1] // 2 - 314:vis.shape[1] // 2 + 314,
            ::-1,
        ]
        vis = cv2.resize(vis, (512, 512))
        plt.clear()
        plt.close()

    label = (
        f"{rbox[0]} {rbox[1]} {rbox[2]} {rbox[3]} {rbox[4]} "
        f"{direction[0]} {direction[1]} {direction[2]}"
    )
    return slice_uint8, label, vis


def parse_args():
    parser = argparse.ArgumentParser(description="Convert short-axis NIfTI volumes into synthetic 2D slices.")
    parser.add_argument("--input-path", type=str, default="ACDC/database/training")
    parser.add_argument("--output-path", type=str, default="data/cmr-3d-ood/converted/trainval")
    parser.add_argument("--label-value", type=int, default=3)
    parser.add_argument("--num-samples-per-case", type=int, default=20)
    parser.add_argument("--slice-height", type=int, default=256)
    parser.add_argument("--slice-width", type=int, default=256)
    parser.add_argument("--max-angle-deg", type=float, default=70.0)
    parser.add_argument("--center-jitter-mm", type=float, default=10.0)
    parser.add_argument("--max-sampling-attempts-factor", type=int, default=100)
    parser.add_argument("--filename-contains", type=str, default=None)
    parser.add_argument("--multiply-affine-by-zooms", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main(args):
    os.makedirs(os.path.join(args.output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "labels"), exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_path, "vis"), exist_ok=True)

    for root, _, files in os.walk(args.input_path):
        for fname in files:
            if not fname.endswith(".nii.gz") or fname.endswith("_gt.nii.gz") or fname.endswith("_4d.nii.gz"):
                continue
            if args.filename_contains is not None and args.filename_contains not in fname:
                continue

            full_path = os.path.join(root, fname)
            gt_path = full_path.replace(".nii.gz", "_gt.nii.gz")
            if not os.path.exists(gt_path):
                print(f"WARNING: skipping {full_path}; missing label file {gt_path}")
                continue

            print(full_path)
            nii = nib.load(full_path)
            gt = nib.load(gt_path)

            start_xyz, end_xyz = long_axis_endpoints_from_nifti(
                gt,
                label_value=args.label_value,
                multiply_affine_by_zooms=args.multiply_affine_by_zooms,
            )
            if start_xyz[2] > end_xyz[2]:
                start_xyz, end_xyz = end_xyz, start_xyz
            bas_ras = np.array(start_xyz)
            tip_ras = np.array(end_xyz)
            center_ras = (bas_ras + tip_ras) / 2

            idx = 0
            attempts = 0
            skipped_empty = 0
            max_attempts = max(args.num_samples_per_case, 1) * args.max_sampling_attempts_factor
            while idx < args.num_samples_per_case:
                attempts += 1
                if attempts > max_attempts:
                    print(
                        f"WARNING: only generated {idx}/{args.num_samples_per_case} samples for {fname} "
                        f"after {max_attempts} attempts; skipped {skipped_empty} empty slices."
                    )
                    break
                try:
                    point_ras = center_ras + args.center_jitter_mm * (np.random.random(center_ras.shape) - 0.5)
                    image, label, vis = get_one_slice(
                        nii,
                        gt,
                        point_ras,
                        bas_ras,
                        tip_ras,
                        label_value=args.label_value,
                        slice_size=(args.slice_height, args.slice_width),
                        max_angle_deg=args.max_angle_deg,
                        multiply_affine_by_zooms=args.multiply_affine_by_zooms,
                        visualize=args.visualize,
                    )
                    stem = f"{fname[:-7]}-{idx:04d}"
                    cv2.imwrite(os.path.join(args.output_path, f"images/{stem}.png"), image)
                    with open(os.path.join(args.output_path, f"labels/{stem}.txt"), "w") as f:
                        f.write(label)
                    if vis is not None:
                        cv2.imwrite(os.path.join(args.output_path, f"vis/{stem}.png"), vis)
                    idx += 1
                except EmptySliceMaskError:
                    skipped_empty += 1
                    continue


if __name__ == "__main__":
    main(parse_args())
