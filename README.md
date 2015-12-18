HMD-Calibration Toolbox for MATLAB(R)
===============

A correct spatial registration of Optical See-Through Head-Mounted Displays (OST-HMD) w.r.t. a user’s eye(s) is an essential problem for any AR application using the such HMDs.

![](https://cloud.githubusercontent.com/assets/7195124/2751076/cf17edce-c8ae-11e3-983a-78ec9d46345a.png)

This toolbox aims to provide core functionalities of the OST-HMD calibration 
including eye localization-based methods and Direct Linear Transform,
and to share the evaluation scheme we used for our experiments. 

## How to use it:
Requirements: MATLAB (with Statistics Toolbox)

At the root directory of this reporisory on your matlab console, just type,
```matlab
>> main
```
then you will see some calibration results like the following:
![](https://cloud.githubusercontent.com/assets/7195124/2751006/7dfb5c80-c8ab-11e3-8d7a-5259f4475f70.png)


If you want to use the core functionality of this tool box for your own calibration, 
please consult the following function files:
```matlab
>> % Functions that give you 3x4 projection matrix
>>
>> % Eye position-based calibration (Full/Recycle Setups)
>> % for Interaction-free Display CAlibration (INDICA) method.
>> P = INDICA_Full   (R_WS, R_WT, t_WT, t_ET, t_WS, ax, ay, w, h);
>> P = INDICA_Recycle(R_WS, R_WT, t_WT, t_ET, t_WS_z, K_E0, t_WE0);
>>
>> % A basic Direct Linear Transform for SPAAM
>> P = DLT(uv,xyz); 
```
The figure below visualizes spatial relationship of each input arguments for INDICA:
![](https://cloud.githubusercontent.com/assets/7195124/2751032/c1a727f6-c8ac-11e3-876c-29d922fad475.png)

Our code is tested on MATLAB(R) version 2013b, and might require some toolboxes.

## Reference:
Please refer to the following publication, which introduces the INDICA method:
```latex
@article{itoh2014-3dui
  author    = {Itoh, Yuta and Klinker, Gudrun},
  title     = {Interaction-Free Calibration for Optical See-Through 
               Head-Mounted Displays based on 3D Eye Localization},
  booktitle = {{Proceedings of the 9th IEEE Symposium on 3D User Interfaces (3D UI)}},
  month     = {march},
  pages     = {75-82},
  year      = {2014}
}
```
![](https://cloud.githubusercontent.com/assets/7195124/2751064/f47f8960-c8ad-11e3-81d0-3bae09c6222b.png)

Note that the evalation scheme in the code is different from the one used in the above paper.
Following the change, a new dataset was acquired and is included in this repository.

## Licence
This repository is provided under MIT license.
