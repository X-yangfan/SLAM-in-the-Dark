from __future__ import division

import torch
import torch.nn.functional as F

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)
    ones = torch.ones(1, h, w).type_as(depth)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert all(condition), f"wrong size for {input_name}, expected {'x'.join(expected)}, got {list(input.size())}"


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat
    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1
    Y_norm = 2 * (Y / Z) / (h - 1) - 1

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1
    ).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1
    ).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1
    ).reshape(B, 3, 3)

    return xmat @ ymat @ zmat


def quat2mat(quat):
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode="euler"):
    translation = vec[:, :3].unsqueeze(-1)
    rot = vec[:, 3:]
    if rotation_mode == "euler":
        rot_mat = euler2mat(rot)
    elif rotation_mode == "quat":
        rot_mat = quat2mat(rot)
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
    transform_mat = torch.cat([rot_mat, translation], dim=2)
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode="euler", padding_mode="zeros"):
    check_sizes(img, "img", "B3HW")
    check_sizes(depth, "depth", "BHW")
    check_sizes(pose, "pose", "B6")
    check_sizes(intrinsics, "intrinsics", "B33")

    cam_coords = pixel2cam(depth, intrinsics.inverse())
    pose_mat = pose_vec2mat(pose, rotation_mode)
    proj_cam_to_src_pixel = intrinsics @ pose_mat
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr, padding_mode)
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    return projected_img, valid_points


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat
    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1
    Y_norm = 2 * (Y / Z) / (h - 1) - 1
    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode="zeros"):
    check_sizes(img, "img", "B3HW")
    check_sizes(depth, "depth", "B1HW")
    check_sizes(ref_depth, "ref_depth", "B1HW")
    check_sizes(pose, "pose", "B6")
    check_sizes(intrinsics, "intrinsics", "B33")

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())
    pose_mat = pose_vec2mat(pose)
    proj_cam_to_src_pixel = intrinsics @ pose_mat
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode)
    valid_mask = (src_pixel_coords.abs().max(dim=-1)[0] <= 1).unsqueeze(1).float()
    return projected_img, valid_mask, projected_depth, computed_depth

