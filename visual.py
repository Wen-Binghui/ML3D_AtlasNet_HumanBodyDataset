from data_loader import Data_set_body
import torch, utils, trimesh
from model.model import EncoderDecoder
import pymesh
from PIL import Image
import numpy as np
import options
"""
Visulize the inference of one input image

"""



option = options.Animals_normal_Option()
mode = 'val'
train_Data = Data_set_body(option.number_points, mode, "animals")
if torch.cuda.is_available():
    option.device = torch.device(f"cuda:0")
else:
    option.device = torch.device(f"cpu")


model = EncoderDecoder(option)
print('NN loaded.')
model_dict_file = 'runs/model_animals_normal_07-31_02h58m.ckpt'
model.load_state_dict(torch.load(model_dict_file))
ind = -1 # index of input image

id = ind if ind>=0 else len(train_Data)+ind
input = train_Data[ind]['img'].unsqueeze(0).float().to(option.device)
img = train_Data[ind]['img'].numpy().transpose(1,2,0)
im=Image.fromarray(np.uint8(img))
im.save("runs/generated_mesh/"+model_dict_file.split('/')[-1].replace('.ckpt',f'_{mode}_{id}.png'))

model.eval()
mesh = model.generate_mesh(input)
pymesh.save_mesh("runs/generated_mesh/"+model_dict_file.split('/')[-1].replace('.ckpt',f'_{mode}_{id}.obj'),\
     mesh, ascii=True)
print('mesh saved')
mesh = trimesh.Trimesh(vertices = mesh.vertices, faces = mesh.faces, process = False)
trimesh.repair.fix_normals(mesh)
mesh.show()

# Uncomment the following to see the points cloud created by the network.
# utils.show_point_cloud(model(input).squeeze(0).view(1,-1,3))

