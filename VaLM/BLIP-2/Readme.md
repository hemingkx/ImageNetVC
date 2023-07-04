# PLM Evaluation

This folder contains all the code used to reimplement our BLIP-2 evaluation.

## Requirements

- Python >= 3.7
- Pytorch >= 1.8.0

## Installation

Please follow the original guideline in [BLIP-2](https://github.com/salesforce/LAVIS) to settle down the environment.

```
conda create -n blip2_ivc python=3.7
pip install -r requirements.txt
```

## Image Sources

We release the top-10 ranked images as well as ImageNet-1K images for re-implementation.

ImageNet-1K images: [Google Drive Link](https://drive.google.com/file/d/1MWnFk1zpYf__NxDBlnASyQm09_omblcV/view?usp=sharing)

Top-10 ranked images: [Baidu Pan Link](https://pan.baidu.com/s/1HlMMXuM1h3OARJY1JzfGwA?pwd=cgs2) (password: cgs2)


## Evaluation

Please replace the files with the original files in [BLIP-2](https://github.com/salesforce/LAVIS).

For details, please check  `ImageNetVC.py`.

```
python ./ImageNetVC.py
```

## Note

This code is based on BLIP-2 [(https://github.com/salesforce/LAVIS/tree/main/projects/blip2)](https://github.com/salesforce/LAVIS/tree/main/projects/blip2). 
