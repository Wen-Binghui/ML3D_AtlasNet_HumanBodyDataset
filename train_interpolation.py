from data_loader import Data_set_body
import torch
from model.model import EncoderDecoder
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import torch.optim as optim
import options
import time
import train

"""
Train a overfitting model using a small dataset created by create_dataset_overfit.py

"""

option_dict = {'animals': options.Animals_Option(),\
    'headposes': options.Headpose_Option(),\
    'humanbody': options.Animals_Option()}

def train_interpolation(dataset_type, model_output):
    option = option_dict[dataset_type]
    train_Data = Data_set_body(option.number_points, 'overfit', dataset_type)
    trainloader = torch.utils.data.DataLoader(train_Data, batch_size=option.batch_size, shuffle=True)
    if torch.cuda.is_available():
        option.device = torch.device(f"cuda:0")
    else:
        option.device = torch.device(f"cpu")
    chamferDist = chamfer_3DDist()

    model = EncoderDecoder(option)
    opt = optim.Adam(model.parameters(), lr=option.lrate)

    train.train(model, chamferDist, opt, trainloader, trainloader, option, model_output)

if __name__ == "__main__":
    time = time.strftime("%m-%d_%Hh%Mm", time.localtime())
    dataset_type = 'humanbody'
    output_file = f'runs/model_humanbody_interp_{time}.ckpt'
    train_interpolation(dataset_type, output_file)