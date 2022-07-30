import numpy as np
import torch
from pathlib import Path
from PIL import Image

class Data_set_body(torch.utils.data.Dataset):

    def __init__(self, num_sample_points, split, dataset_name):
        """
        :param num_sample_points: number of points to sample for sdf values per shape
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit', 'test']
        self.split = split
        self.num_sample_points = num_sample_points
        self.dataset_name = dataset_name
        self.items = Path(f"split_txt/split_txt_{dataset_name}/{split}_split.txt").read_text().splitlines()  # keep track of shape identifiers based on split

    def __getitem__(self, index):
       

        item = self.items[index]
        image_path = 'render/render_'+ self.dataset_name + "/" + self.split + '/' + item
        image=torch.from_numpy(np.copy(Image.open(image_path)).transpose(2,0,1)).float()
        points_path = 'pointcloud/pc_'+ self.dataset_name + "/"  + self.split + '/' + item.split('_')[0] + '.npy'
        npy = torch.from_numpy(np.load(points_path))
        rand_index = np.random.choice(np.arange(npy.shape[0]), self.num_sample_points, replace = False)
        pos_tensor = npy[rand_index, :]



        return {'img': image,\
            'points': pos_tensor}


    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.items)


    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['img'] = batch['img'].to(device)
