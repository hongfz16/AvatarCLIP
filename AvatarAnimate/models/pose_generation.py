from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Tuple
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


def pose_padding(pose):
    assert pose.shape[-1] == 69 or pose.shape[-1] == 63
    if pose.shape[-1] == 63:
        padded_zeros = torch.zeros_like(pose)[..., :6]
        pose = torch.cat((pose, padded_zeros), dim=-1)
    return pose


class BasePoseGenerator(nn.Module, metaclass=ABCMeta):
    """
    Base class for pose generation
    """
    def __init__(self, name: str,
                       topk: Optional[int] = 5,
                       smpl_path: Optional[str] = '../smpl_models',
                       vposer_path: Optional[str] = 'data/vposer'):
        super().__init__()
        self.name = name
        self.topk = topk
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
    def get_topk_poses(self,
                       text: str):
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

    def calculate_pose_score(self, text: str, pose: Tensor) -> float:
        text_feature = self.get_text_feature(text).unsqueeze(0)
        pose_feature = self.get_pose_feature(pose)
        score = F.cosine_similarity(text_feature, pose_feature).item()
        return float(score)

    def sort_poses_by_score(self, text, poses):
        poses.sort(key=lambda x: self.calculate_pose_score(text, x), reverse=True)
        return poses


class PoseOptimizer(BasePoseGenerator):
    """
    This method will directly optimize SMPL theta with the guidance from CLIP
    """
    def __init__(self, 
                 optim_name: Optional[str] = 'Adam',
                 optim_cfg: Optional[dict] = {'lr': 0.01},
                 num_iteration: Optional[int] = 500,
                 **kwargs):
        super().__init__(**kwargs)
        self.optim_name = optim_name
        self.optim_cfg = optim_cfg
        self.num_iteration = num_iteration
    
    def get_pose(self, text_feature: Tensor) -> Tensor:
        pose = nn.Parameter(torch.randn(63))
        cls = getattr(torch.optim, self.optim_name)
        optimizer = cls([pose], **self.optim_cfg)
        for i in tqdm(range(self.num_iteration)):
            new_pose = pose.to(self.device)
            clip_feature = self.get_pose_feature(new_pose).squeeze(0)
            loss = 1 - F.cosine_similarity(clip_feature, text_feature, dim=-1)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return pose_padding(pose.data).to(self.device)

    def get_topk_poses(self, text: str) -> Tensor:
        text_feature = self.get_text_feature(text)
        poses = [self.get_pose(text_feature) for _ in range(self.topk)]
        poses = self.sort_poses_by_score(text, poses)
        poses = torch.stack(poses, dim=0)
        return poses


class VPoserOptimizer(BasePoseGenerator):
    """
    This method will optimize SMPL theta in the latent space of VPoser
    """
    def __init__(self,
                 optim_name: Optional[str] = 'Adam',
                 optim_cfg: Optional[dict] = {'lr': 0.01},
                 num_iteration: Optional[int] = 500,
                 **kwargs):
        super().__init__(**kwargs)
        self.optim_name = optim_name
        self.optim_cfg = optim_cfg
        self.num_iteration = num_iteration
    
    def get_pose(self, text_feature: Tensor) -> Tensor:
        latent_code = nn.Parameter(torch.randn(32))
        cls = getattr(torch.optim, self.optim_name)
        optimizer = cls([latent_code], **self.optim_cfg)
        for i in tqdm(range(self.num_iteration)):
            new_latent_code = latent_code.to(self.device).unsqueeze(0)
            new_pose = self.vp.decode(new_latent_code)['pose_body']
            new_pose = new_pose.contiguous().view(-1)
            clip_feature = self.get_pose_feature(new_pose).squeeze(0)
            loss = 1 - F.cosine_similarity(clip_feature, text_feature, dim=-1)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return pose_padding(new_pose.detach())

    def get_topk_poses(self, text: str) -> Tensor:
        text_feature = self.get_text_feature(text)
        poses = [self.get_pose(text_feature) for _ in range(self.topk)]
        poses = self.sort_poses_by_score(text, poses)
        poses = torch.stack(poses, dim=0)
        return poses


class VPoserRealNVP(BasePoseGenerator):
    """
    This method will generate SMPL theta from a pretrained conditional RealNVP.
    The code is adapted from https://github.com/senya-ashukha/real-nvp-pytorch

    `dim` is the dimension of both input and output. (for vposer is 32)
    `hdim` is the dimension of hidden layer.
    """
    def __init__(self,
                 dim: Optional[int] = 32,
                 hdim: Optional[int] = 256,
                 num_block: Optional[int] = 8,
                 num_sample: Optional[int] = 10,
                 num_batch: Optional[int] = 50,
                 ckpt_path: Optional[str] = 'data/pose_realnvp.pth',
                 **kwargs):
        
        super().__init__(**kwargs)
        self.prior = distributions.MultivariateNormal(
            torch.zeros(dim).to(self.device), torch.eye(dim).to(self.device))
        self.dim = dim
        self.num_sample = num_sample
        self.num_batch = num_batch
        self.s = nn.ModuleList()
        self.t = nn.ModuleList()
        self.num_block = num_block
        mask = torch.randn(num_block, 1, dim)
        mask[mask > 0] = 1
        mask[mask < 0] = 0
        self.register_buffer('mask', mask)
        for i in range(num_block):
            self.s.append(
                nn.Sequential(
                    nn.Linear(dim + 512, hdim),  # concat clip feature
                    nn.LeakyReLU(),
                    nn.Linear(hdim, hdim),
                    nn.LeakyReLU(),
                    nn.Linear(hdim, dim),
                    nn.Tanh()
                )
            )
            self.t.append(
                nn.Sequential(
                    nn.Linear(dim + 512, hdim),
                    nn.LeakyReLU(),
                    nn.Linear(hdim, hdim),
                    nn.LeakyReLU(),
                    nn.Linear(hdim, dim)
                )
            )
        data = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(data['state_dict'], strict=False)
        self.s = self.s.to(self.device)
        self.t = self.t.to(self.device)
        self.mask = self.mask.to(self.device)
        self.eval()

    def decode(self, x: Tensor, features: Tensor) -> Tensor:
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            trans = torch.cat((x_, features), dim=-1)
            s = self.s[i](trans) * (1 - self.mask[i])
            t = self.t[i](trans) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def sample(self, bs: int, features: Tensor) -> Tensor:
        z = self.prior.sample((bs, 1)).squeeze(1).to(self.device)
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        features = features.repeat(bs, 1)
        x = self.decode(z, features)
        return x

    def encode(self, x: Tensor, features: Tensor):
        """
        This is only used during training.
        """
        log_det = torch.zeros(x.shape[0]).type_as(x)
        z = x
        for i in reversed(range(self.num_block)):
            z_ = self.mask[i] * z
            trans = torch.cat((z_, features), dim=-1)
            s = self.s[i](trans) * (1 - self.mask[i])
            t = self.t[i](trans) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det -= s.sum(dim=1)
        return z, log_det

    def get_pose(self, text_feature: Tensor) -> Tensor:
        text_feature = text_feature.unsqueeze(0)
        best_score = 0
        with torch.no_grad():
            for i in tqdm(range(self.num_batch)):
                latent_codes = self.sample(self.num_sample, text_feature)
                poses = self.vp.decode(latent_codes)['pose_body'].reshape(self.num_sample, -1)
                pose_feature = self.get_pose_feature(poses)
                score = F.cosine_similarity(pose_feature, text_feature)
                idx = torch.argmax(score)
                if score[idx] > best_score:
                    best_pose = poses[idx]
                    best_score = score[idx]
        return best_pose

    def get_topk_poses(self, text: str) -> Tensor:
        text_feature = self.get_text_feature(text)
        poses = [self.get_pose(text_feature) for _ in range(self.topk)]
        poses = self.sort_poses_by_score(text, poses)
        poses = torch.stack(poses, dim=0)
        return poses


class VPoserCodebook(BasePoseGenerator):
    """
    This method will find out the poses which are most similar with given text from a codebook.
    """
    def __init__(self,
                 codebook_path='data/codebook.pth',
                 pre_topk=40,
                 filter_threshold=0.07,
                 **kwargs):
        super().__init__(**kwargs)
        data = torch.load(codebook_path)
        self.codebook = data['codebook'].to(self.device)
        self.codebook_embedding = data['codebook_embedding'].to(self.device)
        self.pre_topk = pre_topk
        self.filter_threshold = filter_threshold

    def suppress_duplicated_poses(self, poses: Tensor, threshold: float) -> Tensor:
        new_poses = []
        for pose in poses:
            if len(new_poses) == 0:
                new_poses.append(pose)
            else:
                min_dis = 10
                for j in range(len(new_poses)):
                    cur_dis = torch.abs(pose - new_poses[j]).mean()
                    min_dis = min(cur_dis, min_dis)
                if min_dis > threshold:
                    new_poses.append(pose)
        poses = torch.stack(new_poses, dim=0)
        return poses

    def get_topk_poses(self, text: str) -> Tensor:
        with torch.no_grad():
            text_feature = self.get_text_feature(text).unsqueeze(0)
            score = F.cosine_similarity(
                self.codebook_embedding, text_feature).view(-1)
            _, indexs = torch.topk(score, self.pre_topk)
            latent_codes = self.codebook[indexs]
            poses = self.vp.decode(latent_codes)['pose_body'].reshape(self.pre_topk, -1)
            poses = self.suppress_duplicated_poses(poses, threshold=self.filter_threshold)
            poses = poses[:self.topk]
        return poses
