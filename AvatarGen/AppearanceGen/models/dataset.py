import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json
import imageio
import logging
from scipy import ndimage
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class SMPL_Dataset:
    def __init__(self, conf):
        super(SMPL_Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        with open(os.path.join(self.data_dir, 'transforms_train.json'), 'r') as fp:
            meta = json.load(fp)
        
        self.images = []
        self.poses = []
        self.images_lis = []
        
        for frame in meta['frames']:
            fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
            self.images.append(imageio.imread(fname))
            self.images_lis.append(fname)
            self.poses.append(np.array(frame['transform_matrix']))
        
        self.n_images = len(self.images)
        self.images = (np.array(self.images) / 255.).astype(np.float32)
        self.images = self.images[:, :, ::-1]
        self.images = torch.from_numpy(self.images.copy()).cpu()
        self.masks = torch.zeros_like(self.images)
        self.masks[self.images != 0] = 1.0

        self.poses = np.array(self.poses).astype(np.float32)
        self.poses = torch.from_numpy(self.poses).to(self.device)

        self.H, self.W = self.images[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)
        self.render_poses = torch.stack([pose_spherical(90, angle, 2.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        self.image_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        self.K = np.array([
            [self.focal, 0,          0.5*self.W],
            [0,          self.focal, 0.5*self.H],
            [0,          0,          1         ]
        ])
        self.K = torch.from_numpy(self.K).cpu()

        print('Load data: End')
    
    def gen_rays_silhouettes(self, pose, max_ray_num, mask):
        if mask.sum() == 0:
            return self.gen_rays_pose(pose, resolution_level=4)
        struct = ndimage.generate_binary_structure(2, 2)
        dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=10).astype(np.int32)
        current_ratio = dilated_mask.sum() / float(mask.shape[0] * mask.shape[1])
        W = H = min(self.H, int(np.sqrt(max_ray_num / current_ratio)))
        tx = torch.linspace(0, self.W - 1, W)
        ty = torch.linspace(0, self.H - 1, H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        resized_dilated_mask = torch.nn.functional.interpolate(
            torch.from_numpy(dilated_mask).reshape(256, 256, 1).permute(2, 0, 1).unsqueeze(0).float(), size=(H, W)).squeeze()
        masked_rays_v = rays_v[resized_dilated_mask > 0]
        masked_rays_o = rays_o[resized_dilated_mask > 0]

        return masked_rays_o, masked_rays_v, W, resized_dilated_mask > 0

    def gen_rays_pose(self, pose, resolution_level=1):
        """
        Generate rays at world space given pose.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # rays_v = torch.matmul(self.poses[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * self.poses[img_idx, :3, :3], -1)
        rays_o = self.poses[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        # rays_v = torch.matmul(self.poses[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_v = torch.sum(rays_v[..., None, :] * self.poses[img_idx, :3, :3], -1)
        rays_o = self.poses[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def near_far_from_sphere(self, rays_o, rays_d, is_sphere=False):
        # if not is_sphere:
        #     return 0.5, 3
        # else:
        #     return 0.5, 1
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1
        near[near < 0] = 0
        far = mid + 1
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        img = img[:, ::-1, :]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
