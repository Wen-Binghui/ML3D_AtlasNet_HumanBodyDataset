import numpy as np
import os
import utils

### Main program
np.random.seed(0)

total_lim = 20
limit = {'train': total_lim*0.7, \
        'test': total_lim*0.15, \
        'val': total_lim*0.15, \
        'overfit': 3}

split_list = ['train','test','val','overfit']
for split in split_list:
    if not os.path.exists("render/"+split+'/'):
        os.mkdir("render/"+split+'/')
    if not os.path.exists("pointcloud/"+split+'/'):
        os.mkdir("pointcloud/"+split+'/')
if not os.path.exists("split_txt/"):
        os.mkdir("split_txt/")

obj_file_list = os.listdir("mesh")
n_obj = len(obj_file_list)
index_train = np.random.choice(np.arange(n_obj), int(0.7 * n_obj), replace=False)
index_test_and_val = np.setdiff1d(np.arange(n_obj), index_train, True)
index_val = index_test_and_val[np.random.choice(np.arange(index_test_and_val.size), int(0.5*index_test_and_val.size), replace=False)]
index_test = np.setdiff1d(index_test_and_val, index_val)
index_overfit = np.random.choice(np.arange(n_obj), int(0.001 * n_obj), replace=False)
rot_z=[-110, -90, -60, -30, -10]
rot_x=[30, 45, 60, 75, 90]

index={'train': index_train, 'test': index_test, 'val': index_val, 'overfit': index_overfit}
# index={'train': index_train, 'test': index_test, 'val': index_val, 'overfit': index_overfit}
for split in split_list:
    count = 0 
    with open('split_txt/' + split+'_split.txt','w') as f:
        for i in index[split]:
            file = obj_file_list[i]
            target_prefix_render = "render/" +split+'/' + file.replace(".obj", "_")
            target_prefix_cloud = "pointcloud/" + split + '/' + file.replace(".obj", ".npy")
            utils.gen_pointclouds("mesh/" + file, target_prefix_cloud)
            for iz in range(len(rot_z)):
                for ix in range(len(rot_x)):
                    file_name = target_prefix_render + str(iz) + "_" + str(ix) + ".png"
                    utils.gen_rendering("mesh/" + file, file_name, rot_z[iz], rot_x[ix])
                    f.write(file.replace(".obj", "_") + str(iz) + "_" + str(ix) + ".png"+'\n')
            count+=1
            print(split + '  ' + str(count))
            if count>limit[split]:
                break