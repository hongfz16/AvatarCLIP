import torch
from torch import Tensor
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import smplx
import pyrender
import trimesh
from body_visualizer.tools.vis_tools import imagearray2file
from human_body_prior.body_model.body_model import BodyModel


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code borrowed from https://github.com/nkolot/SPIN
    """
    def __init__(self, focal_length=5000, img_res=512):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res,
            viewport_height=img_res,
            point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        
        smpl_path = '../smpl_models'
        smpl = smplx.create(smpl_path, 'smpl')
        self.faces = smpl.faces

    def __call__(self, vertices, background_color=(255, 255, 255)):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=(0.4, 0.4, 0.4, 1.0))

        camera_translation = np.array([0., 0., 25.])

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length, fy=self.focal_length,
            cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32)
        valid_mask = (rend_depth > 0)[:,:,None]
        background = np.ones_like(color[:, :, :3]) * np.array(background_color).reshape((1, 1, 3))
        output_img = color[:, :, :3] * valid_mask + background * (1 - valid_mask)
        output_img = output_img.astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        return output_img


render = Renderer()


def render_smpl_params(poses: Tensor):
    assert len(poses.shape) == 2 and poses.shape[-1] == 69
    smpl_path = '../smpl_models'
    smpl = smplx.create(smpl_path, 'smpl').to(poses.device)
    bs = poses.shape[0]
    global_orient = torch.zeros(bs, 3).type_as(poses)
    global_orient[:, 0] = np.pi
    output = smpl(body_pose=poses, global_orient=global_orient)
    vertices = output.vertices.detach().cpu().numpy()
    image_list = [render(v) for v in vertices]
    images = np.stack(image_list, axis=0)
    return images


def render_pose(pose: Tensor, image_path: str):
    assert len(pose.shape) == 1
    if pose.shape[-1] == 72:
        pose = pose[3:]
    elif pose.shape[-1] == 63:
        padding_zeros = torch.zeros_like(pose)[:6]
        pose = torch.cat((pose, padding_zeros), dim=-1)

    pose = pose.detach()
    device = pose.device
    
    pose = pose.unsqueeze(0)
    images = render_smpl_params(pose)
    images = images.reshape(1, 1, 1, 512, 512, 3)
    img = imagearray2file(images, outpath=image_path)


def render_motion(motion: Tensor, video_path: str):
    assert len(motion.shape) == 2
    if motion.shape[-1] == 72:
        motion = motion[:, 3:]
    elif motion.shape[-1] == 63:
        padding_zeros = torch.zeros_like(motion)[:, :6]
        motion = torch.cat((motion, padded_zeros), dim=-1)
    device = motion.device
    images = render_smpl_params(motion)
    num_frame = motion.shape[0]
    images = images.reshape(1, 1, num_frame, 512, 512, 3)
    img = imagearray2file(images, outpath=video_path)
