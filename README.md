# PCL: Proxy-based Contrastive Learning for Domain Generalization (CVPR'22)

Official PyTorch implementation of [PCL: Proxy-based Contrastive Learning in Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/papers/Yao_PCL_Proxy-Based_Contrastive_Learning_for_Domain_Generalization_CVPR_2022_paper.pdf).

Xufeng Yao,  Yang Bai,  Xinyun Zhang, Yuechen Zhang, Qi Sun, Ran Chen, Ruiyu Li, Bei Yu

Note that this project is built upon [SWAD](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414) and [DomainBed@3fe9d7](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414).

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for our study.

```
Python: 3.8.6
PyTorch: 1.7.0+cu92
Torchvision: 0.8.1+cu92
CUDA: 9.2
CUDNN: 7603
NumPy: 1.19.4
PIL: 8.0.1
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
# OfficeHome
python train_all.py OH0 --dataset OfficeHome --deterministic \
--trial_seed 0 --steps 3000 --checkpoint_freq 300 --data_dir your_data_dir

# PACS
python train_all.py PACS0 --dataset PACS --deterministic \
--trial_seed 0 --checkpoint_freq 300 --steps 5000 --data_dir your_data_dir

# TerraIncognita
python train_all.py TR0 --dataset TerraIncognita --deterministic \
--trial_seed 0 --checkpoint_freq 1000 --steps 5000 --data_dir your_data_dir
```

Experiment results are reported as a table. In the table, the row `SWAD` indicates out-of-domain accuracy from SWAD.
The row `SWAD (inD)` indicates in-domain validation accuracy.

Example results:

```
# OfficeHome
+------------+---------+---------+---------+------------+---------+
| Selection  |   Art   | Clipart | Product | Real_World |   Avg.  |
+------------+---------+---------+---------+------------+---------+
|   oracle   | 64.882% | 55.842% | 76.267% |  78.600%   | 68.898% |
|    iid     | 64.882% | 53.265% | 76.267% |  78.371%   | 68.196% |
|    last    | 60.968% | 50.773% | 74.240% |  76.363%   | 65.586% |
| last (inD) | 82.440% | 80.166% | 76.457% |  79.601%   | 79.666% |
| iid (inD)  | 83.816% | 84.081% | 79.826% |  81.133%   | 82.214% |
|    SWAD    | 67.302% | 58.763% | 79.448% |  81.440%   | 71.738% |
| SWAD (inD) | 86.022% | 86.993% | 82.424% |  83.130%   | 84.642% |
+------------+---------+---------+---------+------------+---------+
```

## Citation

```
@inproceedings{yao2022pcl,
  title={PCL: Proxy-based Contrastive Learning for Domain Generalization},
  author={Yao, Xufeng and Bai, Yang and Zhang, Xinyun and Zhang, Yuechen and Sun, Qi and Chen, Ran and Li, Ruiyu and Yu, Bei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7097--7107},
  year={2022}
}
```


