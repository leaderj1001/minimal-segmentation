# minimal-segmentation
  - DeepLabV3+ (WORK IN PROCESS)
  - Reference
    - [Paper](https://arxiv.org/pdf/1802.02611.pdf)
    - Author: Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
    - Organization: Google
    
## Usage
  1. Download Cityscapes Dataset ([Cityscapes Link](https://www.cityscapes-dataset.com/))
  2. Data Tree
  ```
  Dataset
   ├── prepare.py
   │   ├── cityscapes
   │   │   ├── leftImg8bit_trainvaltest
   |   |   |   ├── leftImg8bit
   │   │   │   |   ├── train
   │   │   │   |   ├── val
   │   │   │   |   ├── test
   │   │   ├── gtFine_trainvaltest
   |   |   |   ├── gtFine
   │   │   │   |   ├── train
   │   │   │   |   ├── val
   │   │   │   |   ├── test
  ```
  3. Train
  ```
  python main.py --evaluation False
  ```
  4. Test
  ```
  python main.py --evaluation True
  ```

## Experiment

| Datasets | Model | Mean IoU |
| :---: | :---: | :---: |
Cityscape | DeepLabV3+ | 76.074 |

## Todo
  - Distributed
  - COCO dataset
