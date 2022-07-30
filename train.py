from data_loader import Data_set_body
import torch
from model.model import EncoderDecoder
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import torch.optim as optim
import options
import time
option_dict = {'animals': options.Animals_Option(),\
    'headposes': options.Headpose_normal_Option()}

def train(model, loss_criterion, optimizer, trainloader, valloader, option, output_file):
    best_loss = 10000000000

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
                    print(f'better loss, model saved. loss:{loss_total_val}')
                    torch.save(model.state_dict(), output_file) # model_best.ckpt
                    best_loss = loss_total_val

                # set model back to train
                model.train()


def train_normal(dataset_type, model_output):
    option = option_dict[dataset_type]
    train_Data = Data_set_body(option.number_points, 'train', dataset_type)
    val_Data = Data_set_body(option.number_points, 'val', dataset_type)
    trainloader = torch.utils.data.DataLoader(train_Data, batch_size=option.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_Data, batch_size=option.batch_size, shuffle=False)
    if torch.cuda.is_available():
        option.device = torch.device(f"cuda:0")
    else:
        option.device = torch.device(f"cpu")
    chamferDist = chamfer_3DDist()

    model = EncoderDecoder(option)
    opt = optim.Adam(model.parameters(), lr=option.lrate)

    train(model, chamferDist, opt, trainloader, valloader, option, model_output)

if __name__ == "__main__":
    time = time.strftime("%m-%d_%Hh%Mm", time.localtime())
    dataset_type = 'headposes'
    output_file = f'runs/model_headposes_normal_{time}.ckpt'
    train_normal(dataset_type, output_file)