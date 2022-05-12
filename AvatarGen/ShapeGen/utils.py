import os
import torch
import numpy as np
import neural_renderer as nr

obj_fname = './data/smpl_uv.obj'
_, _, texture = nr.load_obj(obj_fname, load_texture=True, texture_size=8)

def render_one_batch(v, f):
    camera_distance = 2
    elevation = 0
    texture_size = 8

    batch_size = v.shape[0]

    vertices = v
    faces = torch.from_numpy(f.astype(np.int32)).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    textures = texture.reshape(1, faces.shape[1], texture_size, texture_size, texture_size, 3).cuda().repeat(vertices.shape[0], 1, 1, 1, 1, 1)

    renderer = nr.Renderer(camera_mode='look_at').cuda()
    images_list = []
    for angle in range(150, 160, 10):
        renderer.eye = nr.get_points_from_angles(camera_distance, 0, angle)
        images, _, _ = renderer(vertices, faces, textures)
        images_list.append(images)
        
    renderer.eye = nr.get_points_from_angles(camera_distance, 0, 170)
    steady_image, _, _ = renderer(vertices, faces, textures)
    steady_image = (steady_image.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    
    images = torch.cat(images_list, 0)

    detached_images = (images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    return images, detached_images, steady_image

def writeOBJ(file, V, F, Vt=None, Ft=None):
    if not Vt is None:
        assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'
        
    with open(file, 'w') as file:
        # Vertices
        for v in V:
            line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'
            file.write(line)
        # UV verts
        if not Vt is None:
            for v in Vt:
                line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'
                file.write(line)
        # 3D Faces / UV faces
        if Ft:
            F = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(F,Ft)]
        else:
            F = [[str(i + 1) for i in f] for f in F]        
        for f in F:
            line = 'f ' + ' '.join(f) + '\n'
            file.write(line)

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
