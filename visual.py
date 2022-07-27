from data_loader import Data_set_body
import torch, utils
from model.model import EncoderDecoder
import pymesh


option = utils.Option()
train_Data = Data_set_body(option.number_points, 'overfit', "headposes")
if torch.cuda.is_available():
    option.device = torch.device(f"cuda:0")
else:
    option.device = torch.device(f"cpu")


model = EncoderDecoder(option)

model_dict_file = 'runs/model_best_headposes3.ckpt'
model.load_state_dict(torch.load(model_dict_file))

input = train_Data[-1]['img'].unsqueeze(0).float().to(option.device)

model.eval()
mesh = model.generate_mesh(input)
pymesh.save_mesh("runs/generated_mesh/"+model_dict_file.split('/')[-1].replace('.ckpt','.obj'),\
     mesh, ascii=True)

utils.show_point_cloud(model(input).squeeze(0).view(1,-1,3))

