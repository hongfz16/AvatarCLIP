import os
import sys
import math
import torch
import imageio
import argparse

import numpy as np
import neural_renderer as nr
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.nn import functional as F
from smplx.lbs import batch_rodrigues
from smplx import build_layer
from tqdm import tqdm
import clip
from PIL import ImageFile, Image

from utils import render_one_batch, writeOBJ

class LinearVAE(nn.Module):
    def __init__(self, in_dim, latent_dim, v_template):
        super(LinearVAE, self).__init__()
        
        self.v_template = v_template
        self.latent_dim = latent_dim

        # encoder
        self.enc1 = nn.Linear(in_features=in_dim, out_features=8192)
        self.enc2 = nn.Linear(in_features=8192, out_features=latent_dim*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=latent_dim, out_features=8192)
        self.dec2 = nn.Linear(in_features=8192, out_features=in_dim)
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        latent_param = self.enc2(self.enc1(x))
        latent_param = latent_param.view(-1, 2, self.latent_dim)
        
        mu = latent_param[:, 0, :]
        log_var = latent_param[:, 1, :]
        
        z = self.reparameterize(mu, log_var)
 
        output = self.dec2(self.dec1(z))
        return output, mu, log_var
    
    def sample(self):
        zgen = torch.tensor(np.random.normal(0., 1., size=(1, self.latent_dim))).float().cuda()
        return self.dec2(self.dec1(zgen)).reshape(-1, 6890, 3) + self.v_template.reshape(1, 6890, 3), zgen
    
    def sample_z(self):
        zgen = torch.tensor(np.random.normal(0., 1., size=(1, self.latent_dim))).float().cuda()
        return zgen
    
    def decode(self, latent):
        return self.dec2(self.dec1(latent)).reshape(-1, 6890, 3) + self.v_template.reshape(1, 6890, 3)

def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

def create_load_AE(in_dim, latent_dim, v_template, pth_fname):
    model_AE = LinearVAE(in_dim, latent_dim, v_template)
    ret = model_AE.load_state_dict(torch.load(pth_fname, map_location='cpu'))
    model_AE = model_AE.eval().requires_grad_(False).cuda()
    return model_AE

def load_clip():
    perceptor, preprocess = clip.load('ViT-B/32', jit=False)
    perceptor = perceptor.eval().requires_grad_(False).cuda()
    return perceptor

def load_codebook(fname):
    codebook_dict = torch.load(fname)
    for i in codebook_dict.keys():
        clip_codebook = codebook_dict[i].cuda()
        break
    return i.cuda(), clip_codebook

def shape_gen(smpl_args, AE_path_fname, codebook_fname, neutral_txt, target_txt):
    smpl_model = build_layer(smpl_args['model_folder'], smpl_args['model_type'],
                             gender = smpl_args['gender'], num_betas = smpl_args['num_betas']).cuda()
    model_AE = create_load_AE(6890 * 3, 16, smpl_model.v_template, AE_path_fname)
    perceptor = load_clip()
    codebook, clip_codebook = load_codebook(codebook_fname)

    ntxt, nweight, nstop = parse_prompt(neutral_txt)
    ttxt, tweight, tstop = parse_prompt(target_txt)
    nembed = perceptor.encode_text(clip.tokenize(ntxt).cuda()).float()
    tembed = perceptor.encode_text(clip.tokenize(ttxt).cuda()).float()
    delta_embed = tembed - nembed
    delta_embed = delta_embed.squeeze(0)

    _beta_zero = torch.zeros([1, 16]).cuda()
    v = model_AE.decode(_beta_zero.detach())
    f = smpl_model.faces
    zero_beta_v = v.clone()
    images, detached_images, steady_image = render_one_batch(v, f)
    images = F.interpolate(images, size=224)
    images -= torch.from_numpy(np.array([0.48145466, 0.4578275, 0.40821073])).reshape(1, 3, 1, 1).cuda()
    images /= torch.from_numpy(np.array([0.26862954, 0.26130258, 0.27577711])).reshape(1, 3, 1, 1).cuda()
    neutral_image_embed = perceptor.encode_image(images).float().mean(0)
    
    cos_dist = (F.normalize(clip_codebook - neutral_image_embed, dim=1) * \
            F.normalize(delta_embed.reshape(-1), dim=0)).sum(-1).reshape(-1)
    best_one = cos_dist.argmax()
    
    v = model_AE.decode(codebook[best_one].reshape(1, 16)).detach().cpu().numpy()
    f = smpl_model.faces
    return v.reshape(-1, 3), f, zero_beta_v.detach().cpu().numpy().reshape(-1, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_model_folder', type=str, default='../../smpl_models')
    parser.add_argument('--AE_path_fname', type=str, default='./data/model_VAE_16.pth')
    parser.add_argument('--codebook_fname', type=str, default='./data/codebook.pth')
    parser.add_argument('--neutral_txt', type=str, default='a 3d rendering of a person in unreal engine')
    parser.add_argument('--target_txt', type=str, default='a 3d rendering of a strong man in unreal engine')
    parser.add_argument('--output_folder', type=str, default='./output/coarse_shape')
    args = parser.parse_args()

    smpl_args = {
        'model_folder': args.smpl_model_folder,
        'model_type': 'smpl',
        'gender': 'neutral',
        'num_betas': 10
    }

    print("Start generating coarse body shape given the target text: {}".format(args.target_txt))
    v, f, zero_beta_v = shape_gen(smpl_args, args.AE_path_fname, args.codebook_fname, \
                     args.neutral_txt, args.target_txt)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    output_fname = os.path.join(args.output_folder, '_'.join(args.target_txt.split(' ')) + '.obj')
    writeOBJ(output_fname, v, f)
    print("Results saved in {}".format(output_fname))
