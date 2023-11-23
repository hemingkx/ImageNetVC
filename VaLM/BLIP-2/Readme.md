# PLM Evaluation

This folder contains all the code used to reimplement our BLIP-2 evaluation.

## Requirements

- Python >= 3.7
- Pytorch >= 1.8.0

## Installation

Please follow the original guideline in [BLIP-2](https://github.com/salesforce/LAVIS) to settle down the environment.

```
conda create -n blip2_ivc python=3.7
cd LAVIS
pip install -e .
```

## Image Sources

We release the top-10 ranked images as well as ImageNet-1K images for re-implementation.

ImageNet-1K images: [Google Drive Link](https://drive.google.com/file/d/1nwI1BlAWpKnXwIRh8J1tvxFJfcNtEZ6B/view?usp=sharing)

Top-10 ranked images: [Google Drive Link](https://drive.google.com/file/d/1CTr68pkiH8cg55sAX1flbsFvZw86ZUC0/view?usp=sharing)


## Evaluation

Please replace the files with the original files in [BLIP-2](https://github.com/salesforce/LAVIS).

Run the script with the following command:

```
CUDA_VISIBLE_DEVICES=0 python ImageNetVC.py --model-name blip2_opt --model-type pretrain_opt2.7b --image-type rank --use-icl
```

For details, please check  `ImageNetVC.py`.

## Note

This code is based on BLIP-2 [(https://github.com/salesforce/LAVIS/tree/main/projects/blip2)](https://github.com/salesforce/LAVIS/tree/main/projects/blip2). 
