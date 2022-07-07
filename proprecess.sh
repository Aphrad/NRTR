## Preprocess for Gold166
mkdir ./H5dataset && mkdir ./H5dataset/Gold166
python ./tools/Gold166/preprocess.py
python ./tools/Gold166/generate_h5dataset.py

## Preprocess for VISoR-40
mkdir ./H5dataset/VISoR-40
python ./tools/VISoR-40/generate_h5dataset.py