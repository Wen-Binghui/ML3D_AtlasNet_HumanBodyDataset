import numpy as np
import pyrender
import trimesh, torch
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def show_point_cloud(torch_tensor):
    pointcloud_np = torch_tensor.squeeze(0).to('cpu').detach().numpy() \
        if len(torch_tensor.shape)==3 else torch_tensor.to('cpu').detach().numpy()
    assert pointcloud_np.shape[1]==3, "pointcloud dim 1 must be 3 but have{}".format(pointcloud_np.shape)
    for i in range(3): print("maximal value in dim {} is {:.2f} and minimal {}".format(i, np.max(pointcloud_np[:,i]), np.min(pointcloud_np[:,i])))
    cld = trimesh.points.PointCloud(pointcloud_np)
    cld.show()

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