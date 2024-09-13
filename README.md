# Style-Adaptive Detection Transformer for Single-Source Domain Generalized Object Detection

By Jianhong Han, Liang Chen and Yupei Wang.

This repository contains the implementation accompanying our paper Style-Adaptive Detection Transformer for Single-Source Domain Generalized Object Detection.


<!---
If you find it helpful for your research, please consider citing:
```
@article{han2024datr,
  title={DATR: Unsupervised Domain Adaptive Detection Transformer with Dataset-Level Adaptation and Prototypical Alignment},
  author={Han, Jianhong and Chen, Liang and Wang, Yupei},
  journal={arXiv preprint arXiv:2405.11765},
  year={2024}
}
```
-->


![](/figs/Figure1.png)

## Acknowledgment
This implementation is bulit upon [DINO](https://github.com/IDEA-Research/DINO/).

## Installation
Please refer to the instructions [here](requirements.txt). We leave our system information for reference.

* OS: Ubuntu 16.04
* Python: 3.10.9
* CUDA: 11.8
* PyTorch: 2.0.1 
* torchvision: 0.15.2

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources.

- Convert the annotation files into COCO-format annotations.

- Modify the dataset path setting within the script [DAcoco.py](./datasets/DAcoco.py) 

```
def build_dayclear(image_set, args):

    #---source domain 训练集
    PATHS_Source = {
        "train": ("",
                  ""),
    }
    #----augmented domain训练集
    PATHS_Target = {
        "train": ("",
                  ""),

    #----source domain 测试集
        "val": ("",
                ""),
```
- All the scenes can be found within the script [__init__.py](./datasets/__init__.py).


## Training / Evaluation / Inference
We provide training script as follows.The settings can be found in the config folder.

- Training with single GPU
```
sh scripts/DINO_train.sh
```
- Training with Multi-GPU
```
sh scripts/DINO_train_dist.sh
```

We provide an evaluation script to evaluate the pre-trained model. --dataset_file is used to specify the test dataset, and --resume is used to specify the path for loading the model.
- Evaluation Model.
```
sh scripts/DINO_eval.sh
```


We provide inference script to visualize detection results. See [inference.py](inference.py) for details
- Inference Model.
```
python inference.py
```

## Pre-trained models
We provide specific [experimental configurations](config/DINO_4scale.py) and [pre-trained model](https://pan.baidu.com/s/1A5iEQTO5nMu90x1W5aw7qA?pwd=1i9z) to facilitate the reproduction of our results. 
You can learn the details of SA-DETR through the paper, and please cite our papers if the code is useful for your papers. Thank you!



Task | mAP50  |
------------| ------------- | 
**Daytime-Clear**  | 64.8% | 
**Dusk-Rainy**  | 45.4% | 
**Night-Rainy**  | 23.0% | 
**Daytime-Foggy**  | 42.6% | 
**Night-Clear**  | 46.0% | 

## Reference
https://github.com/IDEA-Research/DINO
