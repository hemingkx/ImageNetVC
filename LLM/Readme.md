# LLM Evaluation

This folder contains all the code used to reimplement our LLM evaluation on ImageNetVC.

## Requirements

- Python >= 3.7
- Pytorch >= 1.5.0

## Installation

```
conda create -n plm_ivc python=3.7
pip install -r requirements.txt
```

## Evaluation

Please modify the `model_path` in the `init_model` function with your own. Then run the script with the following command:

```
CUDA_VISIBLE_DEVICES=0 python ImageNetVC.py --model-name llama-7b --use-icl
```

For details, please check  `ImageNetVC.py`.
