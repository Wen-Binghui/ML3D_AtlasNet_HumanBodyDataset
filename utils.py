import numpy as np
import pyrender
import trimesh, torch
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]

def gen_rendering(file, target_filename, z_rot, x_rot):
    meshdata = trimesh.load(open(file), file_type = 'obj')
    mesh = pyrender.Mesh.from_trimesh(meshdata)
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0, 0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1,1,1,1], intensity=2e3)
    node_mesh = scene.add(mesh, pose=  np.eye(4))
    node_light = scene.add(light, pose=  np.eye(4))
    t = np.array([0,0,2])
    T = np.eye(4)
    T[0:3, 3] += t 
    R = np.zeros((4, 4))
    R[3, 3] = 1
    R[:3, :3] += Rot.from_euler('ZYX', [z_rot, 0,x_rot], degrees=True).as_matrix()
    pose = R @ T
    node_camera = scene.add(camera, pose = pose)
    Renderer = pyrender.OffscreenRenderer(224,224)
    flags = pyrender.constants.RenderFlags.RGBA
    Rendered_fig, _ = Renderer.render(scene, flags)
    im=Image.fromarray(np.uint8(Rendered_fig)).resize((224,224))
    im.save(target_filename)
    print(target_filename)
    del(Renderer)

def gen_pointclouds(file, target):
    data = trimesh.load(open(file), file_type = 'obj').vertices.view(np.ndarray).astype(np.float32)
    with open(target, 'wb') as f:
        np.save(f, data)