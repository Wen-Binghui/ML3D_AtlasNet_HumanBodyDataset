from data_loader import Data_set_body
import torch
from model.model import EncoderDecoder
import utils
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import torch.optim as optim



option = utils.Option()
train_Data = Data_set_body(option.number_points, 'overfit')
trainloader = torch.utils.data.DataLoader(train_Data, batch_size=option.batch_size, shuffle=True)
if torch.cuda.is_available():
    option.device = torch.device(f"cuda:0")
else:
    option.device = torch.device(f"cpu")
chamferDist = chamfer_3DDist()

model = EncoderDecoder(option)
opt = optim.Adam(model.parameters(), lr=option.lrate)
output_file = 'runs/model_best_mutlti.ckpt'
utils.train(model, chamferDist, opt, trainloader, trainloader, option, output_file)



