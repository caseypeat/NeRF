
for i in {1..2}:
do
    python -m allign.measure.allign_random_error --configs=./allign/configs/config_random.yaml --params input_paths.logdir_a=./logs/synthetic_allign/vine_C1_0/front/* input_paths.model_a=./logs/synthetic_allign/front/*/model/10000.pth input_paths.pointcloud_a=./logs/synthetic_allign/front/*/pointclouds/pcd/10000.pcd input_paths.logdir_b=./logs/synthetic_allign/vine_C1_0/back/* input_paths.model_b=./logs/synthetic_allign/back/*/model/10000.pth input_paths.pointcloud_b=./logs/synthetic_allign/back/*/pointclouds/pcd/10000.pcd
done