import torch
import numpy as np
import neural_renderer as nr


obj_fname = 'data/smpl_uv.obj'
_, _, t = nr.load_obj(obj_fname, load_texture=True, texture_size=8)


def render_one_batch(vertices, faces, angles, device):

    
    camera_distance = 2
    texture_size = 8

    textures = t.reshape(
        1,
        faces.shape[1],
        texture_size,
        texture_size,
        texture_size,
        3).to(device)
    textures = textures.repeat(vertices.shape[0], 1, 1, 1, 1, 1)

    rot_mat = torch.from_numpy(np.array(
        [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
        dtype=np.float32)).to(device)
    vertices = torch.matmul(vertices, rot_mat)

    renderer = nr.Renderer(camera_mode='look_at').to(device)
    images_list = []
    for angle in angles:
        renderer.eye = nr.get_points_from_angles(
            camera_distance, np.random.randn() * 0.3, angle)
        images, _, _ = renderer(vertices, faces, textures)
        images_list.append(images)

    images = torch.cat(images_list, 0)
    return images
