NRTR: Neuron Reconstruction with TRansformer For 3D Optical Microscopy Images
========

PyTorch training and test code for **NRTR** (**N**euron **R**econstruction **TR**ansformer). 

### Python Environment
```
cython==0.29.26
h5py==3.1.0
numpy==1.19.5
tifffile==2020.9.3
torch==1.10.1
```

## Experiment on VISoR-40 Dataset
### Data Preparation

More details are available in ```./dataset/VISoR-40_dataset/README.md```

### GenerateH5File
```
python ./data_processor/VISoR_40/generate_h5dataset.py
```
### Training


## Experiment on Gold166 Dataset