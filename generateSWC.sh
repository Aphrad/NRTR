CUDA_VISIBLE_DEVICES=1 \
python generateSWC.py --distributed\
					  --eval\
					  --backbone="resnet50"\
					  --hidden_dim=192\
					  --val_batch_size=80\
					  --num_queries=100\
					  --crop_size=64\
					  --test_dataset_path="./H5dataset/gold166/Gold166_train_DETR_64_0612.hdf5"\
					  --eval_model_path="./checkpoint/ResNet50_100/checkpoint0199.pth"\
					  --out_swc_dir="./result/0623"
					#   --test_dataset_path="./H5dataset/VISoR_40/VISoR_40_train_DETR_64_0619.hdf5"\
					#   --eval_model_path="./checkpoint/ResNet34_100_VISoR/checkpoint0072.pth"\