import torch
import numpy as np
import neural_renderer as nr
from smplx.lbs import *

def norm_np_arr(arr):
    return arr / np.linalg.norm(arr)

def lookat(eye, at, up):
    zaxis = norm_np_arr(eye - at)
    xaxis = norm_np_arr(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)
    
    viewMatrix = np.array([
        [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
        [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
        [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
        [0       , 0       , 0       , 1                  ]
    ])
    _viewMatrix = np.array([
        [xaxis[0], yaxis[0], zaxis[0], eye[0]],
        [xaxis[1], yaxis[1], zaxis[1], eye[1]],
        [xaxis[2], yaxis[2], zaxis[2], eye[2]],
        [0       , 0       , 0       , 1     ]
    ])
    # viewMatrix = np.linalg.inv(viewMatrix)
    return _viewMatrix

def random_eye_normal():
    camera_distance = np.random.uniform(1, 2)
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.normal(0, np.pi / 3)
    if theta > np.pi / 2 or theta < -np.pi / 2:
        is_front = 0
    else:
        is_front = 1
    return np.array([
        camera_distance * np.sin(theta) * np.cos(phi),
        camera_distance * np.sin(theta) * np.sin(phi),
        camera_distance * np.cos(theta)
    ]), theta, phi, is_front

def random_eye(is_front=None, distance=None, theta_std=None):
    camera_distance = np.random.uniform(1, 2) if distance is None else distance
    phi = np.random.uniform(0, 2 * np.pi)
    if theta_std == None:
        theta_std = np.pi / 6
    theta = np.random.normal(0, theta_std)
    theta = np.clip(theta, -np.pi / 2, np.pi / 2)
    is_front = np.random.choice(2) if is_front is None else is_front
    if is_front == 0:
        theta += np.pi
    return np.array([
        camera_distance * np.sin(theta) * np.cos(phi),
        camera_distance * np.sin(theta) * np.sin(phi),
        camera_distance * np.cos(theta)
    ]), theta, phi, is_front

def sphere_coord(theta, phi, r = 1.0):
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])

def random_at():
    return np.random.normal(
        np.array([0, 0, 0]),
        np.array([0.1, 0.1, 0.1])
    ).clip(-0.3, 0.3)

def batch_rodrigues(rot_vecs, epsilon: float = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def render_one_batch(v, f, eye, at):
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
    renderer = nr.Renderer(camera_mode='look').cuda()
    renderer.eye = eye.float()
    renderer.camera_direction = (at - eye) / torch.norm(at - eye)
    images, _, _ = renderer(vertices, faces, textures)
    detached_images = images.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
    detached_images = detached_images[:, ::-1]
    return detached_images.copy()

def rgb2hsv(input, epsilon=1e-10):
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)

def differentiable_histogram(x, bins=255):

    if len(x.shape) == 4:
        n_samples, n_chns, _, _ = x.shape
    elif len(x.shape) == 2:
        n_samples, n_chns = 1, 1
    else:
        raise AssertionError('The dimension of input tensor should be 2 or 4.')

    min = x.min()
    max = x.max()

    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    delta = (max - min) / bins

    BIN_Table = torch.range(start=0, end=bins, step=1) * delta + min

    for dim in range(1, bins-1, 1):
        h_r = BIN_Table[dim].item()             # h_r
        h_r_sub_1 = BIN_Table[dim - 1].item()   # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
        hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)

    return hist_torch / delta

def my_lbs(v_shaped, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot: bool = True):
    batch_size = pose.shape[0]
    device, dtype = pose.device, pose.dtype

    # Add shape contribution
#     v_shaped = v_template + blend_shapes(betas, shapedirs)

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

def readOBJ(file):
    V, Vt, F, Ft = [], [], [], []
    with open(file, 'r') as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ','').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ','').split(' ')]
            Vt += [v]
        # Face
        elif t.startswith('f '):
            idx = [n.split('/') for n in t.replace('f ','').split(' ')]
            idx = [i for i in idx if i[0]!='']
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
            # UV face
            if '/' in t:
                f = [int(n[1]) - 1 for n in idx]
                Ft += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft: assert len(F) == len(Ft), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
    else: Vt, Ft = None, None
    return V, F, Vt, Ft
