from data_loader import Data_set
import torch
from model.model import EncoderDecoder
import utils
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import torch.optim as optim

class Option(object):
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
    lrate = 0.001
    batch_size = 16
    print_every_n = 1
    validate_every_n = 5
    max_epochs = 200

    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]

option = Option()
train_Data = Data_set(option.number_points, 'overfit')
trainloader = torch.utils.data.DataLoader(train_Data, batch_size=option.batch_size, shuffle=True)
if torch.cuda.is_available():
    option.device = torch.device(f"cuda:0")
else:
    option.device = torch.device(f"cpu")
chamferDist = chamfer_3DDist()

model = EncoderDecoder(option)
opt = optim.Adam(model.parameters(), lr=option.lrate)

utils.train(model, chamferDist, opt, trainloader, trainloader, option)



