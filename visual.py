from data_loader import Data_set
import torch
from model.model import EncoderDecoder



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
    validate_every_n = 10
    max_epochs = 20

    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]

option = Option()
train_Data = Data_set(option.number_points, 'overfit')
if torch.cuda.is_available():
    option.device = torch.device(f"cuda:0")
else:
    option.device = torch.device(f"cpu")


model = EncoderDecoder(option)

model.load_state_dict(torch.load('runs/model_best.ckpt'))

input = train_Data[1]['img'].unsqueeze(0).float().to(option.device)
print(input.shape)
model.generate_mesh(input)
# output = model(input).squeeze(0)
# print(output.shape)
# utils.show_point_cloud(output)


