import numpy as np
import os
import utils

### Main program
np.random.seed(0)

set_name = 'animals'
num_points = 15000

total_lim = 100
limit = {'train': max(total_lim*0.7,1), \
        'test': max(total_lim*0.15,1), \
        'val': max(total_lim*0.15,1), \
        'overfit': 100}

split_list = list(limit.keys())
split_list = ['overfit']


for split in split_list:
    if not os.path.exists("render/render_"+set_name+'/'+split):
        os.makedirs("render/render_"+set_name+'/'+split)
    if not os.path.exists("pointcloud/pc_"+set_name+'/'+split):
        os.makedirs("pointcloud/pc_"+set_name+'/'+split)
if not os.path.exists("split_txt/split_txt_"+set_name):
        os.makedirs("split_txt/split_txt_"+set_name)

obj_file_list = os.listdir("mesh/mesh_"+set_name)   
obj_file_list.sort()
n_obj = len(obj_file_list)

index_train = np.random.choice(np.arange(n_obj), int(0.7 * n_obj), replace=False)
index_test_and_val = np.setdiff1d(np.arange(n_obj), index_train, True)
index_val = index_test_and_val[np.random.choice(np.arange(index_test_and_val.size), int(0.5*index_test_and_val.size), replace=False)]
index_test = np.setdiff1d(index_test_and_val, index_val)
index_overfit = np.array(list(range(n_obj)))

if set_name == "humanbody":
    rot_z=[-110, -90, -60, -30, -10]
    rot_x=[30, 45, 60, 75, 90]
elif set_name == "headposes":
    rot_z=[-10,-2,0,2,10]    #perpendicular to screen
    rot_x=[-10,-2,0,2,10]      #vertical
    # rot_y =[-10,-2,0,2,10]    #horizon
elif set_name == "animals":
    t = np.array([0,0,6])
    rot_z=[-30, -15, 0, 15, 30]    #perpendicular to screen
    rot_x=[35, 45, 60, 75]      #vertical

index={'train': index_train, 'test': index_test, 'val': index_val, 'overfit': index_overfit}
index={'overfit': index_overfit}
for split in split_list:
    count = 0 
    with open('split_txt/split_txt_'+set_name+ '/' + split+'_split.txt','w') as f:
        for i in index[split]:
            file = obj_file_list[i]
            target_prefix_render = "render/render_"+set_name+'/' +split+'/' + file.replace(".obj", "_")
            target_prefix_cloud = "pointcloud/pc_"+set_name+'/' + split + '/' + file.replace(".obj", ".npy")
            utils.gen_pointclouds("mesh/mesh_" + set_name + '/'  + file, target_prefix_cloud, num = num_points)
            for iz in range(len(rot_z)):
                for ix in range(len(rot_x)):
                    file_name = target_prefix_render + str(iz) + "_" + str(ix) + ".png"
                    utils.gen_rendering("mesh/mesh_" + set_name + '/' + file, file_name, rot_z[iz], rot_x[ix], t)
                    f.write(file.replace(".obj", "_") + str(iz) + "_" + str(ix) + ".png"+'\n')
            count+=1
            print(split + '  ' + str(count))
            if count>limit[split]:
                break
