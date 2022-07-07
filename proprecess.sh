## Preprocess for Gold166
mkdir ./H5dataset && mkdir ./H5dataset/Gold166
python ./data_preprocess/Gold166/preprocess.py
python ./data_preprocess/Gold166/generate_h5dataset.py

## Preprocess for VISoR-40
mkdir ./H5dataset/VISoR-40
python ./data_preprocess/VISoR-40/generate_h5dataset.py