# The Gold166 Dataset from BigNeuron Project
All raw images in the <font color="#dd0000">_Gold166 Dataset_</font> are stored as _.v3dpbd_ format, while all annotations are stored as _.swc_ format. The _.v3dpbd_ format images can be readed by [_Vaa3d_](https://github.com/Vaa3D/Vaa3D_Wiki). 

## Dataset Preprocess
Before training <font color="#dd0000">_NRTR_</font>, you should process <font color="#dd0000">_Gold166 Dataset_</font> as follows:

For the raw image and the annonation in every dictionary(e.g. `"./gold166/p_checked7_taiwan_flycirciut/uint8_ChaMARCM-F000006_seg001.lsm_c_3.tif/"`)
* use _Vaa3d_ to convert _.v3dpbd_ to _image.tif_
  * GUI : Click File $\Rightarrow$ Open _.swc_, Click File $\Rightarrow$ Save as _.tif_
  * Ubuntu : `vaa3d -x convert_file_format -f convert_format -i in_file.v3dpbd -o out_file.tif`
* use _Vaa3d_ to convert _.swc_ to _mat_target.tif_
  * GUI : Click Plug\_in $\Rightarrow$ neuron\_utilities $\Rightarrow$ swc\_to\_maskimage\_sphere\_uint $\Rightarrow$ swc\_to\_maskimage, finally saved as _mat\_target.tif_
  * Ubuntu : `vaa3d -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i <inswc_file> [-p <sz0> <sz1> <sz2>] [-o <maskimage.raw>]`
* rename _.swc_ as _swc_target.swc_

## Experiment Configuration
<font color="#dd0000">_NRTR_</font> is trained and test by following sub-dataset.
```python
train_data_dicts =  ['./gold166/p_checked6_zebrafish_horizontal_cells_UW/', 
                     './gold166/p_checked7_taiwan_flycirciut/', 
                     './gold166/p_checked6_mouse_tufts/', 
                     './gold166/p_checked7_janelia_flylight_part1/', 
                     './gold166/p_checked6_silkmoth_utokyo/',
                     './gold166/p_checked6_mouse_culturedcell_Cambridge_in_vivo_2_photon_PAGFP/',
                     './gold166/p_checked6_mouse_RGC_uw/', 
                     './gold166/p_checked6_zebrafish_adult_RGC_UW/']
test_data_dicts = ['./gold166/p_checked6_janelia_flylight_part2/']   
```
## Download

More information about the <font color="#dd0000">_Gold166 Dataset_</font> is available [here](
https://github.com/BigNeuron/BigNeuron-Wiki).

The <font color="#dd0000">_Gold166 Dataset_</font> can be downloaded from [here](
https://zenodo.org/record/168168/files/gold166.zip).