import torch
import open3d as o3d
import numpy as np
from smplx import build_layer
from torch import Tensor
import torch.nn.functional as F


def read_ply(fname):
    mesh = o3d.io.read_triangle_mesh(fname)
    # vertices = np.asarray(mesh.vertices)
    # vertex_colors = np.asarray(mesh.vertex_colors)
    # faces = np.asarray(mesh.triangles)
    return mesh

def simplify_mesh(mesh_in):
    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 256
    mesh_smp = mesh_in.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
    return mesh_smp

def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8) -> Tensor:
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


def init_smpl_model(model_folder):
    model_type = 'smpl'
    gender = 'neutral'
    num_betas = 10
    smpl_model = build_layer(
        model_folder, model_type = model_type,
        gender = gender, num_betas = num_betas
    )
    return smpl_model


def load_template_smpl(smpl_model, pose_fname):
    beta = torch.zeros([1, 10])
    with open(pose_fname, 'rb') as f:
        pose = torch.from_numpy(np.load(f))
    pose_rot = batch_rodrigues(pose.reshape(-1, 3)).reshape(1, 24, 3, 3)
    template_object = smpl_model(
        betas = beta,
        body_pose = pose_rot[:, 1:],
        global_orient = pose_rot[:, 0, :, :].view(1, 1, 3, 3)
    )
    return template_object, pose_rot, beta


def find_nearest_ind(new_vertices, template_object):
    tv = template_object['vertices'].reshape(1, -1, 3).cpu().numpy()
    new_vertices = new_vertices.reshape(-1, 1, 3)
    dist = ((tv - new_vertices) ** 2).sum(-1)
    ind = np.argmin(dist, 1)
    return ind.reshape(-1)

def inv_lbs(smpl_model, vertices, blend_weights, pose, beta):
    v_shaped = smpl_model.v_template + blend_shapes(beta, smpl_model.shapedirs)
    J = vertices2joints(smpl_model.J_regressor, v_shaped)
    J_transformed, A = batch_rigid_transform(pose, J, smpl_model.parents)
    W = blend_weights.unsqueeze(dim=0)
    num_joints = smpl_model.J_regressor.shape[0]
    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

    v_posed_homo = torch.cat([torch.from_numpy(vertices).reshape(1, -1, 3), torch.ones([1, vertices.shape[0], 1])], dim=2).float()
    v_homo = torch.matmul(torch.inverse(T), v_posed_homo.unsqueeze(-1))

    return v_homo[0, :, :3, 0]

def vertices2joints(J_regressor: Tensor, vertices: Tensor) -> Tensor:
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

def blend_shapes(betas: Tensor, shape_disps: Tensor) -> Tensor:
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)
