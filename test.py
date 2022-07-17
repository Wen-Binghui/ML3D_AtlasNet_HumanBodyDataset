from data_loader import Data_set
import numpy as np
import torch
from model.model import EncoderDecoder
from chamfer_distance import ChamferDistance

ds = Data_set(250, 'train')

class Opt(object):
    template_type = "SPHERE"
    bottleneck_size = 1024 
    number_points = 2500
    number_points_eval = 2500
    num_layers = 2
    nb_primitives = 1
    remove_all_batchNorms = 0
    hidden_neurons = 512
    activation = 'relu'
    SVR = True
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]

opt = Opt()

if torch.cuda.is_available():
    opt.device = torch.device(f"cuda:0")
else:
    opt.device = torch.device(f"cpu")

network = EncoderDecoder(opt)
print("Using: {}".format(opt.device))
out = network(ds[0]['img'].unsqueeze(0).float().to(opt.device))
print(out.shape)

true_out = ds[0]['points']


