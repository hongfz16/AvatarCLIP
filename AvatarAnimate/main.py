import os
import argparse
from pyhocon import ConfigFactory

import torch
from models.builder import (
    build_pose_generator,
    build_motion_generator
)

import numpy as np
from visualize import render_pose, render_motion


def main(conf_path):
    with open(conf_path) as f:
        conf_text = f.read()
        f.close()
        conf = ConfigFactory.parse_string(conf_text)
    
    base_exp_dir = conf.get_string('general.base_exp_dir')
    mode = conf.get_string('general.mode')
    text = conf.get_string('general.text')
    if not os.path.exists(base_exp_dir):
        os.makedirs(base_exp_dir)
    pose_generator = build_pose_generator(dict(conf['pose_generator']))
    candidate_poses = pose_generator.get_topk_poses(text)
    N = candidate_poses.shape[0]
    for i in range(N):
        npy_path = os.path.join(base_exp_dir, 'candidate_%d.npy' % i)
        np.save(npy_path, candidate_poses[i].detach().cpu().numpy())
        image_path = os.path.join(base_exp_dir, 'candidate_%d.jpg' % i)
        render_pose(candidate_poses[i], image_path)
    if mode == 'pose':
        exit(0)
    motion_generator = build_motion_generator(dict(conf['motion_generator']))
    motion = motion_generator.get_motion(text, poses=candidate_poses)
    npy_path = os.path.join(base_exp_dir, 'motion.npy')
    np.save(npy_path, motion.detach().cpu().numpy())
    motion_path = os.path.join(base_exp_dir, 'motion.mp4')
    render_motion(motion, motion_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    main(args.conf)
