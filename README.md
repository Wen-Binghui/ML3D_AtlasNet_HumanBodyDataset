# ML3D_AtlasNet_HumanBodyDataset

## requirement:

python 3.7<br>
pytorch 1.7<br>
trimesh<br>
pyrender<br>
scipy

## How to run:
1. If you want to do a interplation to create smooth transformation of mesh models, you should firstly run create_dataset_overfit.py and then run train_interpolation.py, finally use interpolation_test.ipynb to use the model producing the result;
2. If you want to train  a model to convert 2d image to 3d mesh, you should firstly run create_dataset.py and then run train.py;
3. You can see all the outputs include trained network under runs folder;
4. To visullize the mesh produced by network, you can use visual.py;

## DataSet:
1. Humanbody Set: https://graphics.soe.ucsc.edu/data/BodyModels/index.html；
2. Headpose Set: http://people.csail.mit.edu/sumner/research/deftransfer/data/face-poses.zip；
3. Zodiacal Animal Heads Set: from private provider；

## Our Result:
Interpolation of HumanHeadPose:
![28jun_2_exp_cam2](https://user-images.githubusercontent.com/89215484/183676706-21b5eedf-9b81-4ede-8b27-45ba7c04524a.gif)

Interpolation of ZodiacalHead (Sheep to Rabbit):
![30jun_e0_animals_sheep2rab](https://user-images.githubusercontent.com/89215484/183677570-c5252c9a-5b34-46d0-9da6-8dc30dc103b2.gif)
Interpolation of ZodiacalHead (Snack to Chinken):
![30jun_e0_animals_snack2chicken](https://user-images.githubusercontent.com/89215484/183677693-aff396ef-e00f-4ac0-b71e-807548bacc8d.gif)

More Results can be seen under runs folder.
...
