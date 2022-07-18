# %%
from data_loader import Data_set
import torch
from model.model import EncoderDecoder
from chamfer_distance import ChamferDistance
import utils

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
train_DataLoader = Data_set(opt.number_points, 'train')



if torch.cuda.is_available():
    opt.device = torch.device(f"cuda:0")
else:
    opt.device = torch.device(f"cpu")

network = EncoderDecoder(opt)
print("Using: {}".format(opt.device))
out = network(train_DataLoader[0]['img'].unsqueeze(0).float().to(opt.device)).squeeze(0)
print(out.shape)

true_out = train_DataLoader[0]['points'].unsqueeze(0).to(opt.device)
print(true_out.shape)




# %%
chamferDist = ChamferDistance()

dist1, dist2, idx1, idx2 = chamferDist(out, true_out)
loss = (torch.mean(dist1)) + (torch.mean(dist2))

print(loss.item())


# %%
utils.show_point_cloud(out)


