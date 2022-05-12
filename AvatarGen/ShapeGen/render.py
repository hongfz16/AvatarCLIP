import os
import json
import torch
import argparse

import numpy as np
import neural_renderer as nr
import matplotlib.pyplot as plt

from PIL import Image
from smplx.lbs import *
from smplx import build_layer

from utils import readOBJ

def norm_np_arr(arr):
    return arr / np.linalg.norm(arr)

def lookat(eye, target, up):
    zaxis = norm_np_arr(eye - target)
    xaxis = norm_np_arr(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)
    viewMatrix = np.array([
        [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
        [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
        [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
        [0       , 0       , 0       , 1                  ]
    ])
    viewMatrix = np.linalg.inv(viewMatrix)
    return viewMatrix, xaxis, yaxis, zaxis

def render_for_nerf(v, f):
    camera_distance = 2.2
    elevation = 0
    texture_size = 8
    batch_size = v.shape[0]
    vertices = v.clone()
    faces = torch.from_numpy(f.astype(np.int32)).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    textures = torch.ones(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    rot_mat = torch.from_numpy(np.array(
        [[ 1.,  0.,  0.],
         [ 0.,  0., -1.],
         [ 0.,  1.,  0.]], dtype=np.float32)).cuda()
    vertices = torch.matmul(vertices, rot_mat)
    renderer = nr.Renderer(camera_mode='look_at', image_size=256).cuda()
    images_list = []
    transformation_list = []
    for angle in range(0, 360, 20):
        for elevation in range(-60, 60, 20):
            renderer.eye = nr.get_points_from_angles(camera_distance, elevation, angle)
            images, _, _ = renderer(vertices, faces, textures)
            images_list.append(images)
            transformation, xaxis, yaxis, zaxis = lookat(renderer.eye, np.array([0, 0, 0]), np.array([0, 1, 0]))
            transformation_list.append(transformation)
    images = torch.cat(images_list, 0)
    detached_images = (images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    return images, detached_images, transformation_list

def my_lbs(v_shaped, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot: bool = True):
    batch_size = pose.shape[0]
    device, dtype = pose.device, pose.dtype

    # Add shape contribution
    # v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed

def render_coarse_shape(pose, v_shaped, smpl_args, output_dir):
    smpl_model = build_layer(smpl_args['model_folder'], smpl_args['model_type'],
                             gender = smpl_args['gender'], num_betas = smpl_args['num_betas']).cuda()
    pose = torch.from_numpy(pose).cuda()
    pose_rot = batch_rodrigues(pose.reshape(-1, 3)).reshape(1, 24, 3, 3)
    vertices, joints = my_lbs(
        v_shaped, pose_rot, smpl_model.v_template,
        smpl_model.shapedirs, smpl_model.posedirs,
        smpl_model.J_regressor, smpl_model.parents,
        smpl_model.lbs_weights, pose2rot=False
    )
    f = smpl_model.faces
    images, detached_images, transformation_list = render_for_nerf(vertices, f)

    if not os.path.exists(os.path.join(output_dir, 'img')):
        os.makedirs(os.path.join(output_dir, 'img'), exist_ok=True)
    for i in range(images.shape[0]):
        cur_img = (images[i] * 255).permute(1,2,0).type(torch.uint8).detach().cpu().numpy()
        im = Image.fromarray(cur_img)
        im.save(os.path.join(output_dir, 'img/{}.png'.format(str(i).zfill(4))))
    with open(os.path.join(output_dir, 'transforms_train.json'), 'w') as f:
        data = {
            'camera_angle_x': 60 / 180 * np.pi,
            'frames': []
        }
        for i, t in enumerate(transformation_list):
            data['frames'].append({
                'file_path': 'img/{}'.format(str(i).zfill(4)),
                'transform_matrix': t.tolist()
            })
        json.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_model_folder', type=str, default='../../smpl_models')
    parser.add_argument('--coarse_shape_obj', type=str, default=None)
    parser.add_argument('--pose_type', type=str, choices=['stand_pose', 't_pose'], default='stand_pose')
    parser.add_argument('--output_folder', type=str, default='./output/render')
    args = parser.parse_args()

    assert args.coarse_shape_obj is not None
    
    if args.pose_type == 'stand_pose':
        with open('./output/stand_pose.npy', 'rb') as f:
            pose = np.load(f).astype(np.float32)
    elif args.pose_type == 't_pose':
        pose = np.zeros([1, 24, 3], dtype=np.float32)
        pose[:, 0, 0] = np.pi / 2
    else:
        raise NotImplementedError

    v_shaped, _, _, _ = readOBJ(args.coarse_shape_obj)
    v_shaped = torch.from_numpy(v_shaped.astype(np.float32)).reshape(1, -1, 3).cuda()

    smpl_args = {
        'model_folder': args.smpl_model_folder,
        'model_type': 'smpl',
        'gender': 'neutral',
        'num_betas': 10
    }

    print("Begin rendering obj: {}".format(args.coarse_shape_obj))
    render_coarse_shape(pose, v_shaped, smpl_args, args.output_folder)
    print("Renderings written to: {}".format(args.output_folder))
