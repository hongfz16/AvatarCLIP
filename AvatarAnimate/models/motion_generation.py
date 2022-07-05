from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

import numpy as np
import clip
import smplx
from tqdm import tqdm

from .render import render_one_batch
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from .utils import (
    rotation_6d_to_matrix,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d
)


def pose_padding(pose):
    assert pose.shape[-1] == 69 or pose.shape[-1] == 63
    if pose.shape[-1] == 63:
        padded_zeros = torch.zeros_like(pose)[..., :6]
        pose = torch.cat((pose, padded_zeros), dim=-1)
    return pose


class BaseMotionGenerator(nn.Module, metaclass=ABCMeta):
    """
    Base class for motion generation
    """
    def __init__(self, 
                 name: str,
                 num_frame: Optional[int] = 60,
                 smpl_path: Optional[str] = '../smpl_models',
                 vposer_path: Optional[str] = 'data/vposer'):
        super().__init__()
        self.name = name
        self.num_frame = num_frame
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert self.device == "cuda" # neural_render does not support inference on cpu
        self.clip, _ = clip.load('ViT-B/32', self.device)
        self.smpl = smplx.create(smpl_path, 'smpl').to(self.device)
        self.vp, _ = load_model(
            vposer_path,
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True)
        self.vp = self.vp.to(self.device)
        self.clip.eval()
        self.vp.eval()

    @abstractmethod
    def get_motion(self, text: str, poses: Tensor):
        raise NotImplementedError()

    def get_text_feature(self, text: str) -> Tensor:
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(text)
            text_feature = text_features[0]
        return text_feature

    def get_pose_feature(self,
                         pose: Tensor,
                         angles: Optional[Union[Tuple[float], None]] = None) -> Tensor:
        pose = pose_padding(pose)
        if len(pose.shape) == 1:
            pose = pose.unsqueeze(0)
        bs = pose.shape[0]
        # fix the orientation
        global_orient = torch.zeros(bs, 3).type_as(pose)
        global_orient[:, 0] = np.pi / 2
        output = self.smpl(
            body_pose=pose,
            global_orient=global_orient)
        v = output.vertices
        f = self.smpl.faces
        f = torch.from_numpy(f.astype(np.int32)).unsqueeze(0).repeat(bs, 1, 1).to(self.device)
        if angles is None:
            angles = (120, 150, 180, 210, 240)
        images = render_one_batch(v, f, angles, self.device)
        images = F.interpolate(images, size=224)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        images -= torch.from_numpy(mean).reshape(1, 3, 1, 1).to(self.device)
        images /= torch.from_numpy(std).reshape(1, 3, 1, 1).to(self.device)
        num_camera = len(angles)
        image_embed = self.clip.encode_image(images).float().view(num_camera, -1, 512)
        return image_embed.mean(0)


class MotionInterpolation(BaseMotionGenerator):
    """
    This method will interpolate candidate poses to generate a motion
    """
    def __init__(self,
                 anchor_position=(0, 14, 29, 44, 59),
                 **kwargs):
        super().__init__(**kwargs)
        self.anchor_position = anchor_position
        assert anchor_position[0] == 0
        assert anchor_position[-1] == self.num_frame - 1

    def encode_poses(self, poses: Tensor) -> Tensor:
        if poses.shape[-1] == 69:
            poses = poses[:, :63]
        latent_codes = self.vp.encode(poses).mean
        return latent_codes
    
    def decode_poses(self, latent_codes: Tensor) -> Tensor:
        poses = self.vp.decode(latent_codes)['pose_body'].reshape(self.num_frame, 63)
        return poses
    
    def get_motion(self, text: str, poses: Tensor) -> Tensor:
        candidate_latent_codes = self.encode_poses(poses)
        latent_codes = torch.zeros(self.num_frame, 32).type_as(poses)
        latent_codes[0] = candidate_latent_codes[0]
        for i in range(1, len(self.anchor_position)):
            start_code = candidate_latent_codes[i - 1]
            end_code = candidate_latent_codes[i]
            start_frame = self.anchor_position[i - 1]
            end_frame = self.anchor_position[i]
            interval_lengths = end_frame - start_frame
            delta_per_step = (end_code - start_code) / interval_lengths
            for j in range(interval_lengths):
                latent_codes[start_frame + j + 1] = \
                    latent_codes[start_frame + j] + delta_per_step
        motion = self.decode_poses(latent_codes)
        return pose_padding(motion)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2).float()
        div_term = div_term * (-np.log(10000.0) / d_model)
        div_term = torch.exp(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class MotionXTransformerEncoder(nn.Module):
    """
    Motion Encoder, adapted from https://github.com/Mathux/ACTOR
    """
    def __init__(self,
                 seq_len=16,
                 latent_dim=256,
                 output_dim=256,
                 num_heads=4,
                 ff_size=1024,
                 num_layers=8,
                 activation='gelu',
                 dropout=0.1):
        super().__init__()
        self.input_feats = 55 * 6
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(latent_dim)
        self.query = nn.Parameter(torch.randn(1, self.latent_dim))
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=num_layers)
        self.final = nn.Linear(latent_dim, output_dim)

    def forward(self, input):
        B, T = input.shape[:2]
        input = input.view(B, T, -1)
        feature = self.skelEmbedding(input)
        query = self.query.view(1, 1, -1).repeat(B, 1, 1)
        feature = torch.cat((query, feature), dim=1)
        feature = feature.permute(1, 0, 2).contiguous()
        feature = self.pos_encoder(feature)
        feature = self.seqTransEncoder(feature)[0]
        return self.final(feature)


class MotionXTransformerDecoder(nn.Module):
    """
    Motion Decoder, adapted from https://github.com/Mathux/ACTOR
    """
    def __init__(self,
                 seq_len=16,
                 input_dim=256,
                 latent_dim=256,
                 num_heads=4,
                 ff_size=1024,
                 num_layers=8,
                 activation='gelu',
                 dropout=0.1):
        super().__init__()
        if input_dim != latent_dim:
            self.linear = nn.Linear(input_dim, latent_dim)
        else:
            self.linear = nn.Identity()
        self.input_feats = 55 * 6
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.pos_encoder = SinusoidalPositionalEncoding(latent_dim)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer,
            num_layers=num_layers)

        self.final = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, input):
        B = input.shape[0]
        T = self.seq_len
        input = self.linear(input)
        query = self.pos_encoder.pe[:T, :].view(T, 1, -1).repeat(1, B, 1)
        input = input.view(1, B, -1)
        feature = self.seqTransDecoder(tgt=query, memory=input)
        pose = self.final(feature).permute(1, 0, 2).contiguous()
        pose = pose.view(B, T, 55, 6)
        return pose


class MotionOptimizer(BaseMotionGenerator):

    def __init__(self, 
                 latent_dim: Optional[int] = 256,
                 num_layers: Optional[int] = 4,
                 num_heads: Optional[int] = 4,
                 ckpt_path: Optional[str] = 'data/motion_vae.pth',
                 optim_name: Optional[str] = 'Adam',
                 optim_cfg: Optional[dict] = {'lr': 0.01},
                 num_iteration: Optional[int] = 5000,
                 recon_coef: Optional[Tuple[float]] = (1, 0.8, 0.6, 0.4, 0.2),
                 clip_coef: Optional[float] = 0.001,
                 delta_coef: Optional[float] = 0.01,
                 clip_num_part: Optional[int] = 30,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = MotionXTransformerEncoder(
            seq_len=self.num_frame,
            latent_dim=latent_dim,
            output_dim=latent_dim,
            num_heads=num_heads,
            ff_size=latent_dim * 4,
            num_layers=num_layers
        )
        self.decoder = MotionXTransformerDecoder(
            seq_len=self.num_frame,
            input_dim=latent_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=latent_dim * 4,
            num_layers=num_layers,
        )
        data = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(data['state_dict'], strict=False)
        self.latent_dim = latent_dim
        self.optim_name = optim_name
        self.optim_cfg = optim_cfg
        self.num_iteration = num_iteration
        self.recon_coef = recon_coef
        self.clip_coef = clip_coef
        self.delta_coef = delta_coef
        self.clip_num_part = clip_num_part

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.eval()

    def decode(self, latent_code) -> Tensor:
        if len(latent_code.shape) == 1:
            latent_code = latent_code.unsqueeze(0)
        motion_rot6d = self.decoder(latent_code).view(-1, 6)
        motion_rotmat = rotation_6d_to_matrix(motion_rot6d)
        motion_quat = matrix_to_quaternion(motion_rotmat)
        motion = quaternion_to_axis_angle(motion_quat).reshape(-1, 165)
        motion = motion[:, 3: 66].contiguous()
        return motion
        
    def get_motion(self, text: str, poses: Tensor) -> Tensor:
        if poses.shape[-1] == 69:
            poses = poses[..., :63].contiguous()
        text_feature = self.get_text_feature(text).unsqueeze(0)
        latent_code = nn.Parameter(torch.randn(self.latent_dim))
        cls = getattr(torch.optim, self.optim_name)
        optimizer = cls([latent_code], **self.optim_cfg)
        topk = poses.shape[0]
        for i in tqdm(range(self.num_iteration)):
            new_latent_code = latent_code.to(self.device)
            motion = self.decode(new_latent_code)

            loss = 0
            # reconstruction loss
            ori_poses = poses.unsqueeze(1).repeat(1, self.num_frame, 1)
            gen_poses = motion.unsqueeze(0).repeat(topk, 1, 1)
            ori_poses = ori_poses.view(*ori_poses.shape[:-1], 21, 3)
            gen_poses = gen_poses.view(*gen_poses.shape[:-1], 21, 3)
            loss_recon_per_pose = F.mse_loss(
                matrix_to_rotation_6d(axis_angle_to_matrix(gen_poses)),
                matrix_to_rotation_6d(axis_angle_to_matrix(ori_poses)),
                reduction='none').mean(-1).mean(-1)
            value, _ = torch.min(loss_recon_per_pose, dim=1)
            loss_recon = 0
            for j in range(topk):
                loss_recon = loss_recon + value[j] * self.recon_coef[j]
            loss = loss + loss_recon

            # clip loss
            if self.clip_coef > 0:
                st_idx = np.random.randint(self.clip_num_part)
                part_poses = motion[st_idx::self.clip_num_part].contiguous()
                pose_feature = self.get_pose_feature(part_poses, (150,))
                loss_clip_per_pose = 1 - F.cosine_similarity(pose_feature, text_feature)
                loss_clip = 0
                num_part_poses = part_poses.shape[0]
                for j in range(num_part_poses):
                    coef = (st_idx + j * self.clip_num_part) / self.num_frame
                    loss_clip = loss_clip + coef * loss_clip_per_pose[j]
                loss = loss + loss_clip * self.clip_coef

            # delta loss
            if self.delta_coef > 0:
                first_poses = motion[:-1, :].contiguous()
                second_poses = motion[1:, :].contiguous()
                delta_loss = F.mse_loss(first_poses, second_poses)
                loss = loss - delta_loss * self.delta_coef

            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return pose_padding(motion.data).to(self.device)

