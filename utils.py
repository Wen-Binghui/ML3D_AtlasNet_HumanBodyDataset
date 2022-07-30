import numpy as np
import pyrender, re
import trimesh, torch
from scipy.spatial.transform import Rotation as Rot
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm
from scipy.spatial.transform import Rotation

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

def gen_rendering(file, target_filename, z_rot, x_rot, t):
    meshdata = trimesh.load(open(file), file_type = 'obj')
    
    meshdata.vertices -= np.mean(meshdata.vertices, axis = 0)
    meshdata.vertices /= np.std(meshdata.vertices, axis = 0)

    mesh = pyrender.Mesh.from_trimesh(meshdata)
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0, 0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1,1,1,1], intensity=2e3)
    node_mesh = scene.add(mesh, pose=  np.eye(4))
    node_light = scene.add(light, pose=  np.eye(4))
    T = np.eye(4)
    T[0:3, 3] += t 
    R = np.zeros((4, 4))
    R[3, 3] = 1
    R[:3, :3] += Rot.from_euler('ZYX', [z_rot, 0, x_rot], degrees=True).as_matrix()
    pose = R @ T
    node_camera = scene.add(camera, pose = pose)
    Renderer = pyrender.OffscreenRenderer(224,224)
    flags = pyrender.constants.RenderFlags.RGBA
    Rendered_fig, _ = Renderer.render(scene, flags)
    im=Image.fromarray(np.uint8(Rendered_fig)).resize((224,224))
    im.save(target_filename)
    print(target_filename)
    del(Renderer)

def gen_pointclouds(file, target, num = None):
    if num is None:
        data = trimesh.load(open(file), file_type = 'obj').vertices.view(np.ndarray).astype(np.float32)
    else:
        data = trimesh.sample.sample_surface(trimesh.load(open(file), file_type = 'obj'), num)[0].astype(np.float32)
    x_mean = np.mean(data[:,0])
    
    y_mean = np.mean(data[:,1])
    z_mean = np.mean(data[:,2])
    print(x_mean, y_mean, z_mean)
    data[:,0] =  data[:,0] - x_mean
    data[:,1] =  data[:,1] - y_mean
    data[:,2] =  data[:,2] - z_mean
    std = np.std(data)
    data = data / std
    if num is not None:
        assert len(data) == num
    with open(target, 'wb') as f:
        np.save(f, data)

def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


def write_text_to_image(array, text):
    img = Image.fromarray(array)
    img = ImageOps.expand(img, (40, 20, 0, 0), fill=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("exercise_3/util/font/FreeMono.ttf", 14)
    draw.text((10, 10), text, (0, 0, 0), font=font)
    return np.array(img)


def meshes_to_gif(mesh_folder, output_path, fps):
    w_img = 1024
    obj_file_list = os.listdir(mesh_folder)
    obj_file_list.sort(key = lambda x: int(re.findall(r"\d+",x)[0]) )
    r = pyrender.OffscreenRenderer(w_img, w_img)
    image_buffer = [np.zeros((w_img, w_img, 3), dtype=np.uint8) for i in range(len(obj_file_list))]

    for i, mesh_path in enumerate(tqdm(obj_file_list, desc='visualizing')):
        base_mesh = trimesh.load_mesh(mesh_folder+'/'+mesh_path)
        trimesh.repair.fix_normals(base_mesh)
        loc = np.array([0, 0, 0])
        scale = 24
        base_mesh.apply_translation(-loc)
        base_mesh.apply_scale(1 / scale)
        mesh = pyrender.Mesh.from_trimesh(base_mesh)
        camera_rotation = np.eye(4)
        camera_rotation[:3, :3] = Rotation.from_euler('y', 30, degrees=True).as_matrix() @ Rotation.from_euler('x', -20, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1.25])
        camera_pose = camera_rotation @ camera_translation
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
        scene.add(mesh)
        scene.add(camera, pose=camera_pose)
        for n in create_raymond_lights():
            scene.add_node(n, scene.main_camera_node)
        color, depth = r.render(scene)
        image_buffer[i] = write_text_to_image(color, f"{i}")
    clip = ImageSequenceClip(image_buffer, fps=fps)
    clip.write_gif(output_path, verbose=False, logger=None)
