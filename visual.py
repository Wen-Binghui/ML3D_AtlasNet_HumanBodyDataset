from data_loader import Data_set_body
import torch, utils, trimesh
from model.model import EncoderDecoder
import pymesh
from PIL import Image
import numpy as np

option = utils.Option()
train_Data = Data_set_body(option.number_points, 'overfit', "headposes")
if torch.cuda.is_available():
    option.device = torch.device(f"cuda:0")
else:
    option.device = torch.device(f"cpu")


model = EncoderDecoder(option)

model_dict_file = 'runs/model_best_headposes3.ckpt'
model.load_state_dict(torch.load(model_dict_file))
ind = -50

id = ind if ind>=0 else len(train_Data)+ind
input = train_Data[ind]['img'].unsqueeze(0).float().to(option.device)
img = train_Data[ind]['img'].numpy().transpose(1,2,0)
im=Image.fromarray(np.uint8(img))
im.save("runs/generated_mesh/"+model_dict_file.split('/')[-1].replace('.ckpt',f'_{id}.png'))

model.eval()
mesh = model.generate_mesh(input)
pymesh.save_mesh("runs/generated_mesh/"+model_dict_file.split('/')[-1].replace('.ckpt',f'_{id}.obj'),\
     mesh, ascii=True)

mesh = trimesh.Trimesh(vertices = mesh.vertices, faces = mesh.faces, process = False)
mesh.show()

# utils.show_point_cloud(model(input).squeeze(0).view(1,-1,3))

