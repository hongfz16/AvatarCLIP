import os
import time
import logging
import argparse
import random
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.dataset import SMPL_Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.utils import lookat, random_eye, random_at, render_one_batch, batch_rodrigues
from models.utils import sphere_coord, random_eye_normal, rgb2hsv, differentiable_histogram
from models.utils import my_lbs, readOBJ
import clip
from smplx import build_layer
import imageio

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, is_colab=False, conf=None):
        self.device = torch.device('cuda')
        self.conf_path = conf_path

        if is_colab:
            self.conf = conf
        else:
            # Configuration
            f = open(self.conf_path)
            conf_text = f.read()
            f.close()
            self.conf = ConfigFactory.parse_string(conf_text)

        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = SMPL_Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.max_ray_num = self.conf.get_int('train.max_ray_num', default=112 * 112)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        try:
            self.clip_weight = self.conf.get_float('train.clip_weight')
        except:
            self.clip_weight = None
        try:
            self.extra_color = self.conf.get_bool('model.rendering_network.extra_color')
        except:
            self.extra_color = False
        try:
            self.add_no_texture = self.conf.get_bool('train.add_no_texture')
        except:
            self.add_no_texture = False
        try:
            self.texture_cast_light = self.conf.get_bool('train.texture_cast_light')
        except:
            self.texture_cast_light = False
        try:
            self.use_face_prompt = self.conf.get_bool('train.use_face_prompt')
        except:
            self.use_face_prompt = False
        try:
            self.use_back_prompt = self.conf.get_bool('train.use_back_prompt')
        except:
            self.use_back_prompt = False
        try:
            self.use_silhouettes = self.conf.get_bool('train.use_silhouettes')
        except:
            self.use_silhouettes = False
        try:
            self.head_height = self.conf.get_float('train.head_height')
            print("Use head height: {}".format(self.head_height))
        except:
            self.head_height = 0.65
        try:
            self.use_bg_aug = self.conf.get_bool('train.use_bg_aug')
        except:
            self.use_bg_aug = True
        try:
            self.seed = self.conf.get_int('train.seed')
            # Constrain all sources of randomness
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("Fix seed to: {}".format(self.seed))
        except:
            pass

        try:
            self.smpl_model_path = self.conf.get_string('general.smpl_model_path')
        except:
            self.smpl_model_path = '../../smpl_models'

        try:
            self.pose_type = self.conf.get_string('general.pose_type')
            assert self.pose_type in ['stand_pose', 't_pose']
        except:
            self.pose_type = 'stand_pose'

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = None #NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        # params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        try:
            pretrain_pth = self.conf.get_string('train.pretrain')
        except:
            pretrain_pth = None
        if pretrain_pth is not None:
            logging.info('Load pretrain: {}'.format(pretrain_pth))
            self.load_pretrain(pretrain_pth)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def init_clip(self):
        self.perceptor, preprocess = clip.load('ViT-B/32', jit=False)
        self.perceptor = self.perceptor.eval().requires_grad_(False).cuda()
        self.clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            self.clip_normalizer
        ])
        self.resize = transforms.RandomResizedCrop(224, scale=(1, 1))
        self.random_perspective = transforms.RandomPerspective(p=0, distortion_scale=0.5) #p=0.8 might trigger some strange bug of randomperspective

        def add_noise(v, mean, std):
            tmp = torch.zeros_like(v)
            return v + torch.normal(mean=tmp+mean, std=tmp+std)
        
        prompt = self.conf.get_string('clip.prompt')
        print("Prompt: {}".format(prompt))
        prompt_token = clip.tokenize([prompt]).cuda()
        self.encoded_text = self.perceptor.encode_text(prompt_token).detach()
        
        if self.use_face_prompt:
            face_prompt = self.conf.get_string('clip.face_prompt')
            print("Face Prompt: {}".format(face_prompt))
            face_prompt_token = clip.tokenize([face_prompt]).cuda()
            self.encoded_face_text = self.perceptor.encode_text(face_prompt_token).detach()

        if self.use_back_prompt:
            back_prompt = self.conf.get_string('clip.back_prompt')
            print("Back Prompt: {}".format(back_prompt))
            back_prompt_token = clip.tokenize([back_prompt]).cuda()
            self.encoded_back_text = self.perceptor.encode_text(back_prompt_token).detach()

    def init_smpl(self):
        try:
            template_obj_fname = self.conf['dataset.template_obj']
        except:
            template_obj_fname = None

        model_folder = '../../smpl_models'
        model_type = 'smpl'
        gender = 'neutral'
        num_betas = 10
        smpl_model = build_layer(
            model_folder, model_type = model_type,
            gender = gender, num_betas = num_betas).cuda()

        if self.pose_type == 'stand_pose':
            with open('../ShapeGen/output/stand_pose.npy', 'rb') as f:
                new_pose = np.load(f)
        elif self.pose_type == 't_pose':
            new_pose = np.zeros([1, 24, 3])
            new_pose[:, 0, 0] = np.pi / 2
        else:
            raise NotImplementedError

        new_pose = torch.from_numpy(new_pose.astype(np.float32)).cuda()
        pose_rot = batch_rodrigues(new_pose.reshape(-1, 3)).reshape(1, 24, 3, 3)

        if template_obj_fname is not None:
            # v_dict = torch.load(template_obj_fname)
            # v_shaped = v_dict['v'].reshape(1, -1, 3).cuda()
            v_shaped, _, _, _ = readOBJ(template_obj_fname)
            v_shaped = torch.from_numpy(v_shaped.astype(np.float32)).reshape(1, -1, 3).cuda()
            full_pose = pose_rot.reshape(1, -1, 3, 3)
            vertices, joints = my_lbs(
                v_shaped, full_pose, smpl_model.v_template,
                smpl_model.shapedirs, smpl_model.posedirs,
                smpl_model.J_regressor, smpl_model.parents,
                smpl_model.lbs_weights, pose2rot=False,
            )
            self.v = vertices.clone()
        else:
            beta = torch.zeros([1, 10]).cuda()
            so = smpl_model(betas = beta, body_pose = pose_rot[:, 1:], global_orient = pose_rot[:, 0, :, :].view(1, 1, 3, 3))
            self.v = so['vertices'].clone()
            del so
        
        self.f = smpl_model.faces.copy()

    def train_clip(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        idx_counter = 0

        for iter_i in tqdm(range(res_step)):
            if iter_i == 30010:
                break
            if self.use_face_prompt and iter_i % 4 == 0:
                eye, theta, phi, is_front = random_eye(is_front=1, distance=0.4, theta_std=np.pi/12)
                at = np.array([0, self.head_height, 0.3]).astype(np.float32)
                eye = eye.astype(np.float32)
                eye += at
            else:
                # eye, theta, phi, is_front = random_eye()
                eye, theta, phi, is_front = random_eye_normal()
                at = random_at().astype(np.float32)
                eye = eye.astype(np.float32)
                eye += at
            pose = lookat(eye, at, np.array([0, 1, 0]))
            true_rgb = torch.from_numpy(render_one_batch(self.v, self.f, torch.from_numpy(eye).cuda(), torch.from_numpy(at).cuda()))
            ori_mask = torch.zeros_like(true_rgb)
            ori_mask[true_rgb != 0] = 1
            ori_mask = ori_mask[..., 0]

            if self.use_silhouettes:
                rays_o, rays_d, W, dilated_mask = self.dataset.gen_rays_silhouettes(torch.from_numpy(pose).cuda(), self.max_ray_num, ori_mask)
                H = W
                rays_o = rays_o.float()
                rays_d = rays_d.float()
            else:
                rays_o, rays_d = self.dataset.gen_rays_pose(torch.from_numpy(pose).cuda(), 2.25)
                H, W = rays_o.shape[0], rays_o.shape[1]
                rays_o = rays_o.reshape(H * W, 3).float()
                rays_d = rays_d.reshape(H * W, 3).float()

            true_rgb = torch.nn.functional.interpolate(true_rgb.reshape(256, 256, 3).permute(2, 0, 1).unsqueeze(0), \
                                                       size=(H, W)).squeeze(0).permute(1, 2, 0).cuda().reshape(-1, 3)
            mask = torch.zeros_like(true_rgb)
            mask[true_rgb != 0] = 1
            mask = mask[..., :1]

            if self.use_face_prompt and iter_i % 4 == 0:
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d, is_sphere=True)
            else:
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_bg_aug:
                choice_i = np.random.choice(4)
            else:
                choice_i = 3
            if choice_i == 0:
                background_rgb = torch.ones([1, 3])
            elif choice_i == 1:
                gaussian = torch.normal(torch.zeros([H, W, 1]) + 0.5, torch.zeros([H, W, 1]) + 0.2)
                background_rgb = torch.clamp(gaussian, min=0, max=1).reshape(-1, 1)
            elif choice_i == 2:
                chess_board = torch.zeros([H, W, 1]) + 0.2
                chess_length = H // np.random.choice(np.arange(10,20))
                i, j = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='xy')
                div_i, div_j = i // chess_length, j // chess_length
                white_i, white_j = i[(div_i + div_j) % 2 == 0], j[(div_i + div_j) % 2 == 0]
                chess_board[white_i, white_j] = 0.8
                blur_fn = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
                background_rgb = blur_fn(chess_board.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0).reshape(-1, 1)

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            if self.use_silhouettes and (choice_i == 1 or choice_i == 2):
                masked_background_rgb = background_rgb.reshape(H, W, 1)[dilated_mask].reshape(-1, 1)
            else:
                masked_background_rgb = background_rgb

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=masked_background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            extra_color_fine = render_out['extra_color_fine']

            ## cast light
            if self.add_no_texture or self.texture_cast_light:
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                normals = normals.sum(dim=1)
                normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-7)
                
                # light_dir = torch.from_numpy(np.random.randn(3))
                light_dir = sphere_coord(theta + np.random.uniform(-np.pi/4, np.pi/4), phi + np.random.uniform(-np.pi/4, np.pi/4))
                light_dir = torch.from_numpy(light_dir).float()
                rand_light_d = torch.zeros_like(normals).float().to(normals.device) + light_dir.to(normals.device)
                rand_light_d = rand_light_d / (torch.norm(rand_light_d, dim=-1, keepdim=True) + 1e-7)
                
                rand_diffuse_shading = (normals * rand_light_d).sum(-1, keepdim=True).clamp(min=0, max=1)
                rand_diffuse_shading[torch.isnan(rand_diffuse_shading)] = 1.0
                ambience = np.random.uniform(0, 0.2)
                diffuse = 1 - ambience
                rand_shading = ambience + diffuse * rand_diffuse_shading

                rand_shading_rgb = rand_shading.clone()
                rand_shading_rgb = rand_shading_rgb.reshape(-1, 1).repeat(1, 3).float()
                weight_sum = render_out['weight_sum'].reshape(-1)
                # rand_shading_rgb[weight_sum < 0.5] = 0.0
                rand_shading_rgb[weight_sum < 0.5] = extra_color_fine[weight_sum < 0.5]

                l_ratio = 1
                rand_shading = l_ratio * rand_shading + 1 - l_ratio
                rand_shading[weight_sum < 0.5] = 1.0
                texture_shading = (extra_color_fine * rand_shading).clamp(min=0, max=1)

            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            if self.use_silhouettes:
                background = torch.zeros([H, W, 3]).cuda()
                if choice_i == 0:
                    background[:] = 1
                if choice_i == 1 or choice_i == 2:
                    background[~dilated_mask] = background_rgb.reshape(H, W, 1).repeat(1, 1, 3)[~dilated_mask]

                if self.add_no_texture or self.texture_cast_light:
                    full_texture_shading = background.clone()
                    full_texture_shading[dilated_mask] = texture_shading
                    texture_shading = full_texture_shading.reshape(-1, 3)

                    full_rand_shading_rgb = background.clone()
                    full_rand_shading_rgb[dilated_mask] = rand_shading_rgb
                    rand_shading_rgb = full_rand_shading_rgb.reshape(-1, 3)

                full_extra_color_fine = background.clone()
                full_extra_color_fine[dilated_mask] = extra_color_fine
                extra_color_fine = full_extra_color_fine.reshape(-1, 3)

                full_color_fine = torch.zeros([H, W, 3]).cuda()
                full_color_fine[dilated_mask] = color_fine
                color_fine = full_color_fine.reshape(-1, 3)

                full_weight_sum = torch.zeros([H, W, 1]).cuda()
                full_weight_sum[dilated_mask] = weight_sum
                weight_sum = full_weight_sum.reshape(-1, 1)

            # Loss
            ## L1 Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            ## eikonal loss
            eikonal_loss = gradient_error
            ## mask loss
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            ## clip loss
            if self.use_face_prompt and iter_i % 4 == 0:
                current_text_encoding = self.encoded_face_text
                current_no_texture_text_encoding = self.encoded_face_text
            elif self.use_back_prompt and is_front == 0:
                current_text_encoding = self.encoded_back_text
                current_no_texture_text_encoding = self.encoded_back_text
            else:
                current_text_encoding = self.encoded_text
                current_no_texture_text_encoding = self.encoded_text

            if self.texture_cast_light:
                augmented_extra_rgb = self.resize(texture_shading.reshape(H, W, 3).unsqueeze(0).permute(0, 3, 1, 2))
                augmented_extra_rgb = self.clip_normalizer(self.random_perspective(augmented_extra_rgb))
                encoded_renders = self.perceptor.encode_image(augmented_extra_rgb)
                cosine = torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                torch.mean(current_text_encoding, dim=0), dim=0)
            else:
                augmented_extra_rgb = self.resize(extra_color_fine.reshape(H, W, 3).unsqueeze(0).permute(0, 3, 1, 2))
                augmented_extra_rgb = self.clip_normalizer(self.random_perspective(augmented_extra_rgb))
                encoded_renders = self.perceptor.encode_image(augmented_extra_rgb)
                cosine = torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                torch.mean(current_text_encoding, dim=0), dim=0)
            if self.add_no_texture:
                augmented_shading_rgb = self.resize(rand_shading_rgb.reshape(H, W, 3).unsqueeze(0).permute(0, 3, 1, 2))
                augmented_shading_rgb = self.clip_normalizer(self.random_perspective(augmented_shading_rgb))
                encoded_shading_rgb = self.perceptor.encode_image(augmented_shading_rgb)
                cosine_shading = torch.cosine_similarity(torch.mean(encoded_shading_rgb, dim=0),
                                                torch.mean(current_no_texture_text_encoding, dim=0), dim=0)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight +\
                   (1.0 - cosine) * self.clip_weight
                
            if self.add_no_texture:
                loss += (1.0 - cosine_shading) * self.clip_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/cosine', cosine, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image(idx = 58)
                # idx_counter = (idx_counter + 1) % self.dataset.n_images

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        # self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def load_pretrain(self, checkpoint_name):
        # checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'], strict=False)

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            # 'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def render_geometry_cast_light(self):
        # eye, theta, phi, is_front = random_eye(is_front=1, distance=0.4, theta_std=np.pi/12)
        # at = np.array([0, self.head_height, 0.3]).astype(np.float32)
        # eye = eye.astype(np.float32)
        # eye += at

        phi = 0
        theta = 0
        camera_distance = 0.5
        eye = np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])
        at = np.array([0, self.head_height, 0.3])
        eye += at

        # phi = 0
        # theta = 0
        # camera_distance = 1.5
        # eye = np.array([
        #     camera_distance * np.sin(theta) * np.cos(phi),
        #     camera_distance * np.sin(theta) * np.sin(phi),
        #     camera_distance * np.cos(theta)
        # ])
        # at = np.array([0, 0, 0])
        pose = lookat(eye, at, np.array([0, 1, 0]))
        rays_o, rays_d = self.dataset.gen_rays_pose(torch.from_numpy(pose).cuda(), 0.5)
        H, W = rays_o.shape[0], rays_o.shape[1]
        rays_o = rays_o.reshape(H * W, 3).float().split(self.batch_size)
        rays_d = rays_d.reshape(H * W, 3).float().split(self.batch_size)
        out_cast_light = []

        background_rgb = None
        # choice_i = np.random.choice(4)
        choice_i = 3
        counter = -1

        light_dir = sphere_coord(theta + np.random.uniform(-np.pi/4, np.pi/4), phi + np.random.uniform(-np.pi/4, np.pi/4))
        light_dir = torch.from_numpy(light_dir).float()

        rand_number_1 = np.random.choice(np.arange(10,20))
        blur_fn = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            counter += 1
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            if choice_i == 0:
                background_rgb = torch.ones([1, 3])
            elif choice_i == 1:
                gaussian = torch.normal(torch.zeros([self.batch_size, 1]) + 0.5, torch.zeros([self.batch_size, 1]) + 0.2)
                background_rgb = torch.clamp(gaussian, min=0, max=1).reshape(-1, 1)
            elif choice_i == 2:
                chess_board = torch.zeros([H, W, 1]) + 0.2
                chess_length = H // rand_number_1
                i, j = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='xy')
                div_i, div_j = i // chess_length, j // chess_length
                white_i, white_j = i[(div_i + div_j) % 2 == 0], j[(div_i + div_j) % 2 == 0]
                chess_board[white_i, white_j] = 0.8
                
                background_rgb = blur_fn(chess_board.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0).reshape(-1, 1)
                background_rgb = background_rgb[counter * self.batch_size: (counter + 1) * self.batch_size]

            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            color_fine = render_out['color_fine']
            extra_color_fine = render_out['extra_color_fine']

            n_samples = self.renderer.n_samples + self.renderer.n_importance
            normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
            normals = normals.sum(dim=1)
            normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-7)
            
            # light_dir = sphere_coord(theta + np.random.uniform(-np.pi/4, np.pi/4), phi + np.random.uniform(-np.pi/4, np.pi/4))
            # light_dir = torch.from_numpy(light_dir).float()
            rand_light_d = torch.zeros_like(normals).float().to(normals.device) + light_dir.to(normals.device)
            rand_light_d = rand_light_d / (torch.norm(rand_light_d, dim=-1, keepdim=True) + 1e-7)
            
            rand_diffuse_shading = (normals * rand_light_d).sum(-1, keepdim=True).clamp(min=0, max=1)
            rand_diffuse_shading[torch.isnan(rand_diffuse_shading)] = 1.0
            # ambience = np.random.uniform(0, 0.2)
            ambience = 0
            diffuse = 1 - ambience
            rand_shading = ambience + diffuse * rand_diffuse_shading

            rand_shading_rgb = rand_shading.clone()
            rand_shading_rgb = rand_shading_rgb.reshape(-1, 1).repeat(1, 3).float()
            weight_sum = render_out['weight_sum'].reshape(-1)
            # rand_shading_rgb[weight_sum < 0.5] = 0.0
            rand_shading_rgb[weight_sum < 0.5] = extra_color_fine[weight_sum < 0.5]

            l_ratio = 1
            rand_shading = l_ratio * rand_shading + 1 - l_ratio
            rand_shading[weight_sum < 0.5] = 1.0
            texture_shading = (extra_color_fine * rand_shading).clamp(min=0, max=1)

            out_cast_light.append(texture_shading.detach().cpu().numpy())

        cast_light_img = np.concatenate(out_cast_light, 0).reshape(H, W, 3)

        imageio.imwrite(
            os.path.join(self.base_exp_dir, 'cast_light_texture_head_black.png'),
            to8b(cast_light_img)
        )

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_extra_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                if self.extra_color:
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                    out_extra_rgb_fine.append(render_out['extra_color_fine'].detach().cpu().numpy())
                else:
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)
        extra_img_fine = None
        if len(out_extra_rgb_fine) > 0:
            extra_img_fine = (np.concatenate(out_extra_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.poses[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_extra_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_extra_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_extra_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           cv.cvtColor(extra_img_fine[..., i], cv.COLOR_RGB2BGR))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        ### extract color
        pt_vertices = torch.from_numpy(vertices).cuda().reshape(-1, 1, 3).float()
        rays_o_list = [
            np.array([0, 0, 2]),
            np.array([0, 0, -2]),
            np.array([0, 2, 0]),
            np.array([0, -2, 0]),
            np.array([2, 0, 0]),
            np.array([-2, 0, 0]),
        ]
        rgb_final = None
        diff_final = None
        for rays_o in rays_o_list:
            rays_o = torch.from_numpy(rays_o.reshape(1, 3)).repeat(vertices.shape[0], 1).cuda().float()
            rays_d = pt_vertices.reshape(-1, 3) - rays_o
            rays_d = rays_d / torch.norm(rays_d, dim=-1).reshape(-1, 1)
            dist = torch.norm(pt_vertices.reshape(-1, 3) - rays_o, dim=-1)

            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            dist = dist.reshape(-1).split(self.batch_size)
            out_rgb_fine = []
            depth_diff = []
            for i, (rays_o_batch, rays_d_batch) in enumerate(zip(rays_o, rays_d)):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near, far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb)
                if self.extra_color:
                    out_rgb_fine.append(render_out['extra_color_fine'].detach().cpu().numpy())
                else:
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                
                weights = render_out['weights']
                mid_z_vals = render_out['mid_z_vals']
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                depth_batch = (mid_z_vals[:, :n_samples] * weights[:, :n_samples]).sum(dim=1).detach().cpu().numpy()
                dist_batch = dist[i].detach().cpu().numpy()
                depth_diff.append(np.abs(depth_batch - dist_batch))

                del render_out

            out_rgb_fine = np.concatenate(out_rgb_fine, axis=0).reshape(vertices.shape[0], 3)
            depth_diff = np.concatenate(depth_diff, axis=0).reshape(vertices.shape[0])
            
            if rgb_final is None:
                rgb_final = out_rgb_fine.copy()
                diff_final = depth_diff.copy()
            else:
                ind = diff_final > depth_diff
                ind = ind.reshape(-1)
                rgb_final[ind] = out_rgb_fine[ind]
                diff_final[ind] = depth_diff[ind]

        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=to8b(rgb_final))
        trimesh.exchange.export.export_mesh(mesh, os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)), file_type='ply')
        # mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='smpl')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    
    if args.mode == 'validate_mesh' or \
       args.mode == 'render_geometry_cast_light':
        args.is_continue = True
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
        runner.render_geometry_cast_light()
    elif args.mode == 'train_clip':
        runner.init_clip()
        runner.init_smpl()
        runner.train_clip()
    elif args.mode == 'render_geometry_cast_light':
        runner.render_geometry_cast_light()
