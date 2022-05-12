import os
import copy
import torch
import struct
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch import Tensor
from smplx import build_layer
import torch.nn.functional as F

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

def read_ply(fname):
    mesh = o3d.io.read_triangle_mesh(fname)
    # vertices = np.asarray(mesh.vertices)
    # vertex_colors = np.asarray(mesh.vertex_colors)
    # faces = np.asarray(mesh.triangles)
    return mesh

def write_ply(mesh, fname):
    o3d.io.write_triangle_mesh(fname, mesh)

def cleanup_mesh(mesh):
    num_vertices = np.asarray(mesh.vertices).shape[0]
    mesh.compute_adjacency_list()
    adj = np.zeros([num_vertices, 100]).astype(np.int32) - 1
    for i, l in enumerate(mesh.adjacency_list):
        adj[i][:len(l)] = np.array(list(l))
    list_of_island = []
    biggest_island = -1
    biggest_island_num_vertices = -1
    def count_list_of_set(l):
        return sum([len(s) for s in l])
    global_visited = np.zeros([num_vertices])
    while count_list_of_set(list_of_island) != num_vertices:
        visited = set()
        queue = [np.where(global_visited == 0)[0][0]]
        while len(queue) != 0:
            cind = int(queue.pop(0))
            if cind in visited:
                continue
            visited.add(cind)
            global_visited[cind] = 1
            cadj = adj[cind]
            nind = cadj[cadj != -1]
            nind = [i for i in nind if i not in visited]
            queue += nind
        list_of_island.append(visited.copy())
        if len(list_of_island) == 1:
            biggest_island = 0
            biggest_island_num_vertices = len(visited)
        elif len(visited) > biggest_island_num_vertices:
            biggest_island = len(list_of_island) - 1
            biggest_island_num_vertices = len(visited)
    remove_ind = []
    for i, island in enumerate(list_of_island):
        if i == biggest_island:
            continue
        remove_ind += list(island)
    mesh.remove_vertices_by_index(remove_ind)
    return mesh

def init_smpl_model(model_folder):
    # model_folder = '../smplx/models'
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

def lbs(smpl_model, tpose_vertices, blend_weights, pose, beta):
    v_shaped = smpl_model.v_template + blend_shapes(beta, smpl_model.shapedirs)
    J = vertices2joints(smpl_model.J_regressor, v_shaped)
    J_transformed, A = batch_rigid_transform(pose, J, smpl_model.parents)
    W = blend_weights.unsqueeze(dim=0)
    num_joints = smpl_model.J_regressor.shape[0]
    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)
    v_homo = torch.cat([tpose_vertices.reshape(1, -1, 3), torch.ones([1, tpose_vertices.shape[0], 1])], dim=2).float()
    v_posed_homo = torch.matmul(T, v_homo.unsqueeze(-1))

    return v_posed_homo[0, :, :3, 0]

def read_pose_seq(folder):
    dirs = os.listdir(folder)
    dirs = [d for d in dirs if d.startswith('000')]
    dirs = sorted(dirs)
    pose_list = []
    for d in tqdm(dirs):
        pkl_fname = os.path.join(folder, d, 'smpl_param.pkl')
        with open(pkl_fname, 'rb') as f:
            smpl_param_dict = pickle.load(f)
        pose_np = smpl_param_dict['pose'].astype(np.float32)
        rot_poses = torch.from_numpy(pose_np).reshape(-1, 3)
        rot_poses = batch_rodrigues(rot_poses).view(1, 24, 3, 3)
        pose_list.append(rot_poses)
    return pose_list

def read_pose_my(fname):
    poses = np.load(fname)
    #import pdb; pdb.set_trace()
    pose_list = []
    for i in range(poses.shape[0]):
        pose_np = poses[i, :72]
        pose_np[:3] = 0
        pose_np[0] = np.pi / 2
        rot_poses = torch.from_numpy(pose_np).reshape(-1, 3)
        rot_poses = batch_rodrigues(rot_poses).view(1, 24, 3, 3)
        pose_list.append(rot_poses)
    return pose_list

def write_pc2(fname, vertices_list):
    vcount = vertices_list[0].shape[0]
    start_frame = 0
    sample_rate = 60
    num_samples = len(vertices_list)
    fmt = '<12siiffi'
    header = struct.pack(fmt, b'POINTCACHE2\0', 1, vcount, start_frame, sample_rate, num_samples)
    vertices = np.asarray([v.cpu().numpy() for v in vertices_list]).reshape(num_samples, vcount, 3)
    with open(fname, 'wb') as file:
        file.write(header)
        vertices.astype('<f').tofile(file)

#if __name__ == '__main__':
def generate_animation(pose_name):
    tag = 'averyskinnymangeneral'
    name = 'General'
    folder = 'averyskinnyman'
    #pose_name = 'shoot_basketball'

    # mesh = read_ply('exp/smpl/{}_add_no_texture_cast_light_face_back_prompt_silhouettes/meshes/00100000.ply'.format(name))
    mesh = read_ply('exp/smpl/{}/{}/meshes/00029500.ply'.format(folder, name))
    #mesh = read_ply('exp/smpl/{}/{}/meshes/00031500.ply'.format(folder, name))
    ori_vertices = np.asarray(mesh.vertices)
    rot_ori_vertices = np.matmul(
        ori_vertices,
        np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0,-1, 0]], dtype=np.float32)
    )
    mesh.vertices = o3d.utility.Vector3dVector(rot_ori_vertices)
    mesh = cleanup_mesh(mesh)

    target_folder = '/data/text2mesh/for paper/overallresults/{}'.format(tag)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    target_ply_fname = os.path.join(target_folder, '{}_cleaned_apose.ply'.format(name))
    o3d.io.write_triangle_mesh(target_ply_fname, mesh)
    # o3d.visualization.draw_geometries([mesh])

    smpl_model = init_smpl_model('../smplx/models')
    template_object, pose_rot, beta = load_template_smpl(smpl_model, '../stand_pose.npy')
    nearest_ind = find_nearest_ind(np.asarray(mesh.vertices), template_object)

    smpl_blend_weights = smpl_model.lbs_weights
    mesh_blend_weights = torch.gather(smpl_blend_weights, 0, torch.from_numpy(nearest_ind).unsqueeze(-1).repeat(1, smpl_blend_weights.shape[-1]))

    tpose_vertices = inv_lbs(smpl_model, np.asarray(mesh.vertices), mesh_blend_weights, pose_rot, beta)

    tpose_mesh = copy.deepcopy(mesh)
    tpose_mesh.vertices = o3d.utility.Vector3dVector(tpose_vertices)
    # o3d.visualization.draw_geometries([tpose_mesh])

    #pose_folder = '../../Garment4D/dataset/CLOTH3D/CLOTH3D/{}'.format(pose_name)
    #pose_list = read_pose_seq(pose_folder)
    
    pose_fname = '/data/text2mesh/for paper/generated_poses/final_motions/{}/action.npy'.format(pose_name)
    pose_list = read_pose_my(pose_fname)

    vertices_list = []
    for pose_rot in tqdm(pose_list):
        vertices_list.append(lbs(smpl_model, tpose_vertices, mesh_blend_weights, pose_rot, beta))

    target_ply_fname = os.path.join(target_folder, '{}.ply'.format(name))
    target_pc2_fname = os.path.join(target_folder, '{}.pc2'.format(pose_name))

    #o3d.io.write_triangle_mesh(target_ply_fname, tpose_mesh)
    write_pc2(target_pc2_fname, vertices_list)
    
    # new_pose = torch.zeros([1, 72])
    # new_pose[:, 0] = np.pi / 2
    # new_pose[:, 10] = np.pi / 2
    # new_pose_rot = batch_rodrigues(new_pose.reshape(-1, 3)).reshape(1, 24, 3, 3)

    # posed_vertices = lbs(smpl_model, tpose_vertices, mesh_blend_weights, new_pose_rot)
    # posed_mesh = copy.deepcopy(mesh)
    # posed_mesh.vertices = o3d.utility.Vector3dVector(posed_vertices)
    # o3d.visualization.draw_geometries([posed_mesh])

    ## inverse smpl for testing
    # tpose_vertices = inv_lbs(smpl_model, template_object['vertices'].cpu().numpy().squeeze(0), smpl_blend_weights, pose_rot, beta)
    # smpl_mesh = o3d.geometry.TriangleMesh(vertices = o3d.utility.Vector3dVector(template_object['vertices'].cpu().numpy().squeeze(0)), triangles = o3d.utility.Vector3iVector(smpl_model.faces))
    # o3d.visualization.draw_geometries([smpl_mesh, mesh])

if __name__ == '__main__':
    
    motion_names = [
        #'arguing',
        #'bending_over',
        #'crying',
        #'drinking_water',
        #'excited',
        #'kicking_soccer',
        #'praying',
        'raising_both_arms',
        #'running',
        #'shoot_basketball'
    ]
    
    #motion_names = ['sitting']
    for mn in motion_names:
        generate_animation(mn)


