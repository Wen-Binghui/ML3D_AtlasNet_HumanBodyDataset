import numpy as np
import pyrender
import trimesh, torch
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import os
from data_loader import Data_set_body

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Option(object):
    template_type = "SPHERE"
    bottleneck_size = 1024 
    overfit = True
    number_points = 15000
    nb_primitives = 4 # number_points/nb_primitives = nb_pts_in_primitive

    number_points_eval = 15000
    num_layers = 3
    remove_all_batchNorms = 0
    hidden_neurons = 512
    activation = 'relu'
    SVR = True
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    lrate = 0.001
    batch_size = 4
    print_every_n = 1
    validate_every_n = 10
    max_epochs = 200

    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]

class OptionSquare(object):
    template_type = "SQUARE"
    bottleneck_size = 1024 
    overfit = True
    number_points = 15000
    nb_primitives = 50 # number_points/nb_primitives = nb_pts_in_primitive

    number_points_eval = 15000
    num_layers = 2
    remove_all_batchNorms = 0
    hidden_neurons = 512
    activation = 'relu'
    SVR = True
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    lrate = 0.001
    batch_size = 8
    print_every_n = 1
    validate_every_n = 10
    max_epochs = 200

    def __init__(self):
        self.dim_template = self.dim_tem


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
    t = np.array([0,0,25])
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
    data = trimesh.load(open(file), file_type = 'obj').vertices.view(np.ndarray).astype(np.float32)
    with open(target, 'wb') as f:
        np.save(f, data)


def train(model, loss_criterion, optimizer, trainloader, valloader, option, output_file):
    best_loss = 10000000

    loss_criterion.to(option.device)

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of running average of train loss for printing
    train_loss_running = 0.
    for epoch in range(option.max_epochs):
        for i, batch in enumerate(trainloader):
            # move batch to device
            Data_set_body.move_batch_to_device(batch, option.device)
            optimizer.zero_grad()
            prediction = model(batch['img'])
            prediction = prediction.view(prediction.shape[0], -1, 3).contiguous()
            true_out = batch['points'].contiguous()


            dist1, dist2, _, _  = loss_criterion(true_out, prediction)
            loss_total = ((torch.mean(dist1)) + (torch.mean(dist2))).to(option.device)

            loss_total.backward()
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i
            if iteration % option.print_every_n == (option.print_every_n - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / option.print_every_n:.5f}')
                train_loss_running = 0.
            # validation evaluation and logging
            if iteration % option.validate_every_n == (option.validate_every_n - 1):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()
                loss_total_val = 0
                total= 0
                # forward pass and evaluation for entire validation set
                for batch_val in valloader:
                    Data_set_body.move_batch_to_device(batch_val, option.device)

                    with torch.no_grad():
                        prediction = model(batch_val['img'])
                        prediction = prediction.view(prediction.shape[0], -1, 3).contiguous()
                        true_out = batch_val['points']

                        dist1, dist2, _, _  = loss_criterion(prediction, true_out)
                        loss_val_per = ((torch.mean(dist1)) + (torch.mean(dist2))).to(option.device)

                    loss_total_val += loss_val_per.item()
                    total += batch_val['points'].shape[0]

                if loss_total_val < best_loss:
                    print('better loss, model saved.')
                    torch.save(model.state_dict(), output_file) # model_best.ckpt
                    best_loss = loss_total_val

                # set model back to train
                model.train()
