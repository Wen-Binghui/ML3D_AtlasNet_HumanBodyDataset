import numpy as np
import os
import utils

### Main program
np.random.seed(0)

set_name = 'animals'
num_points = 15000

limit = {'overfit': 1}

split_list = list(limit.keys())


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

index_overfit = np.array(list(range(n_obj)))
index={'overfit': index_overfit}


if set_name == "humanbody":
    rot_z=[-110, -90, -60, -30, -10]
    rot_x=[30, 45, 60, 75, 90]
elif set_name == "headposes":
    rot_z=[-10,-2,0,2,10]    #perpendicular to screen
    rot_x=[-10,-2,0,2,10]      #vertical
elif set_name == "animals":
    t = np.array([0,0,6])
    rot_z=[-30, -15, 0, 15, 30]    #perpendicular to screen
    rot_x=[35, 45, 60, 75]      #vertical


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
