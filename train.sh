CUDA_VISIBLE_DEVICES=0 \
python main.py --distributed --epochs=200 --backbone="resnet50" --hidden_dim=192 --batch_size=32 --val_batch_size=120 --num_queries=200 --crop_size=64\
--train_dataset_path="./H5dataset/Gold166/Gold166_train_NRTR.hdf5" --test_dataset_path="./H5dataset/Gold166/Gold166_test_NRTR.hdf5"\--out_checkpoint_dir="./checkpoint/"\