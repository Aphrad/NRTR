CUDA_VISIBLE_DEVICES=0
python generateSWC.py --distributed --eval --backbone="resnet50" --hidden_dim=192 --val_batch_size=80 --num_queries=100 --crop_size=64\
--test_dataset_path="./H5dataset/gold166/Gold166_test_NRTR.hdf5" --eval_model_path="./checkpoint/best_model.pth" --out_swc_dir="./result/"\

python generateConnectivity.py  --in_swc="result/p_checked7_janelia_flylight_part1/err_GMR_57C10_AD_01-1xLwt_attp40_4stop1-f-A01-20110325_3_A1-right_optic_lobe.v3draw.extract_5/swc_target.swc"\
                                --out_swc="result/p_checked7_janelia_flylight_part1/err_GMR_57C10_AD_01-1xLwt_attp40_4stop1-f-A01-20110325_3_A1-right_optic_lobe.v3draw.extract_5/cylinder_swc_target.swc"\