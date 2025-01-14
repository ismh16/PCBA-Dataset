# PCBA-DET Dataset
😊This is a object detection dataset for PCBA defect detection

🥳The source code can be accessed in the pcba-yolo folder  

👍VOC and PCB defect dataset tags in txt format are available in "GET" below  

NOTE!! 
 
PCB dataset is from [https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB) 4

Thanks for their code and dataset. If you need to use the PCB defect dataset please visit their website.
## Get Dataset

You can also get some sample datasets in [Baidu Netdisk](https://pan.baidu.com/s/1XdOV2nQaf4gQ6gUoJwEpFg?pwd=zxs6) or [Google Drive](https://drive.google.com/file/d/1lmMe3o7kZG67zcL2ZbJJFfXPpMLWzV2k/view?usp=sharing)

VOC dataset labels in yolo format in [Google drive](https://drive.google.com/file/d/1T0ogvDWhaGrODb6H5r4KLzpIeFMv5oQO/view?usp=drive_link)

PCB defect dataset labels in yolo format in [Google drive](https://drive.google.com/file/d/1wJ94UB-0KCXy-Ytom7Yf5KwMV4io8Ghw/view?usp=drive_link)

Full PCBA-DET dataset in [Baidu Netdisk](https://pan.baidu.com/s/129Drcfg5XHHLTXZ_LqCNRw?pwd=iio1) code = iio1.

Data augmentation data is available in [Baidu Netdisk](https://pan.baidu.com/s/1cxNSv6g9Uax3Uu9XECYddw?pwd=l3bd) code = l3bd.

Data augmentation using attention-gan(fan scratch category) are available in [Baidu Netdisk](https://pan.baidu.com/s/15sOkQXefRDLk7-n8UEk97Q?pwd=sigi) code = sigi.

## Cite
Please cite our [paper](https://www.nature.com/articles/s41598-024-70176-1) while using the PCBA-DET dataset or related research

[1]Shen, M., Liu, Y., Chen, J. et al. Defect detection of printed circuit board assembly based on YOLOv5. Sci Rep 14, 19287 (2024). https://doi.org/10.1038/s41598-024-70176-1

A recent [paper](https://www.ecice06.com/CN/10.19678/j.issn.1000-3428.0070196) on YOLOv8 improvements should also be consulted.

[2]沈明辉,刘宇杰,陈婧,等.基于改进YOLOv8s轻量化网络的组装电脑主板缺陷检测算法[J/OL].计算机工程,1-14[2025-01-10].https://doi.org/10.19678/j.issn.1000-3428.0070196.

## About
### Size
We have 4,000 images and a total of 2,384 data augmented images!!
### Format
Yolo format, we will update when our paper is published.
### About
We photographed from three different angles: from above, from the side, and from a tilted angle.  

![photo angle](https://github.com/ismh16/PCBA-Dataset/blob/main/img/angle.jpg "angjpg")   

We use [makesense](https://www.makesense.ai/) to label the dataset  

And labeled 8 original defects    

![photo category](https://github.com/ismh16/PCBA-Dataset/blob/main/img/category.jpg "catejpg")     

Distribution of defects in PCBA dataset as follows
|  Category  | Number (defects)  |  Number (images)  |
|  :----:  | :----:  |  :----:  |
| Loose fan screws | 1200 | 500 |
| Missing fan screws | 2300 | 1400 |
| Loose motherboard screws | 3300 | 1000 |
| Missing motherboard screws | 3800 | 1900 |
| Loose fan wiring | 1500 | 1500 |
| Missing fan wiring | 1400 | 1400 |
| Fan scratches | 1300 | 1300 |
| Motherboard scratches | 3300 | 1100 |  

## Code
### Train: [PCBA-YOLO(Based on YOLOv5)](#pcba-yolo) or [Lightweight YOLOv8s](#lightweight-yolov8-model)

#### PCBA-YOLO
You can train your own PCBA detection model with the following code:
```
!python train.py --weights 'path/to/your/pre_trained/model.pt' --cfg 'pcba_yolo.yaml' --data 'mainBoard.yaml' --epochs 300 --batch-size 32
```

You can find the '--weights' parameter file in the `./pcba_yolo/weights/`
|  weight  | model  |
|  :----:  | :----:  |
| pcba_yolo_13.pt | PCBA-YOLO(K=13) |
| pcba_yolo_17.pt | PCBA-YOLO(K=17) |
| pcba_yolo_27.pt | PCBA-YOLO(K=27) |
| replk_yolo.pt | Only replknet |
| sppcspc_yolo.pt | Only sppcspc |
| siou_yolo.pt | Only siou |
| replk_sppcspc_yolo.pt | replknet and sppcspc |
| replk_siou_yolo.pt | replknet and siou |
| sppcspc_siou_yolo.pt | sppcspc and siou |
| yolov5s.pt | YOLOv5s |

You can resize K by changing the parameter in RepLKBlock in  `./models/common.py`  at line 1016 

`self.m = nn.Sequential(*(RepLKBlock(c_, c_, 13, 5, 0.0, False) for _ in range(n)))`

where the third parameter 13 is the size of K


We provide yolov5s model, other models are available at [Google drive](https://drive.google.com/drive/folders/1pBx4lROqzg2e51HER2egHbjnAizgXbE8?usp=drive_link)

You can find the '--cfg' parameter file in the `./pcba_yolo/mdoel/`
|  cfg  | model  |
|  :----:  | :----:  |
| pcba_yolo.yaml | PCBA-YOLO |
| replk_yolo.yaml | Only replknet |
| sppcspc_yolo.yaml | Only sppcspc |
| yolov5s.yaml | YOLOv5s |

#### Lightweight YOLOv8 Model
You can train your own PCBA detection model with running `./train.py` in file `YOLOv8_Lightweight`

You can find the '--cfg' parameter file in the `./ultralytics/cfg/models/v8`
|  cfg  | model  |
|  :----:  | :----:  |
| yolov8-replk-ghost-p2.yaml | Our lightweight model |
| yolov8-ghost-p2.yaml | Without replknet |
| yolov8-replk-p2.yaml | Without ghostnet |
| yolov8-replk-ghost.yaml | Without p2 layer |
| yolov8-replk.yaml | Only replknet |
| yolov8-ghost.yaml | Only ghostnet |
| yolov8-p2.yaml | Only p2 layer |
| yolov8.yaml | YOLOv8 |

Pretrained models are available at [Google drive](https://drive.google.com/file/d/14Aoy6RMQSxu92KAahizqJoCW-PTPJ5Dd/view?usp=sharing)

Where yolov8_replk_ghost_p2_agu.pt is the augmented lightweight model

### Validate
You can validate your detection model with the following code:

```
!python val.py --data 'mainBoard.yaml' --weights 'path/to/your/model.pt' --batch-size 32 
```

### Defect Detection
You can use the following code for defect detection:

```
!python detect.py --weights 'path/to/your/model.pt' --source 'path/to/your/image' --data 'mainBoard.yaml'
```

## License  
For academic research only 

You can contact us with <shenmh16@gmail.com>
