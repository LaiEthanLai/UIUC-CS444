# Assignment 3: Self-supervised and Transfer Learning, Object Detection
## Part 1: Self-supervised and Transfer Learning
The first part of assignment 3 is to train a neural network in a self-supervised manner, then fine-tune it for a downstream task. Specifically, we first train our neural network to classify images based on their rotation degrees. In our setting, there are four classes of rotation, namely 0, 90, 180, and 270. The downstream task is image classification. 

We experiment with different fine-tuning strategies, including fine-tune only the few last layers of a model and the whole model. The performance of these models is compared with models being trained from scratch.  

To train the model and run the experiements, please refer to `assignment3_part1/a3_part1_rotation.ipynb`. 

## Result
#### ResNet-18

##### CIFAR-10
| Setting | Test Acc. |
|---------|-----------|
| Rotation | 78.74% |
|    Pretrained, Fine-tune 'layer4' block and 'fc' layer     |     61.2%      |
|    Train 'layer4' block and 'fc' layer from scratch    |     46.23%      |
|    Pretrained, Fine-tune Entire Model     |     84.2%      |
|    Train the Entire Model from scratch    |     82.45%      |

##### ImageNette
| Setting | Test Acc. |
|---------|-----------|
| Rotation | 65.09% |

#### ResNet-101
| Setting | Test Acc. |
|---------|-----------|
| Rotation | 90.76% |
|    Pretrained, Fine-tune 'layer4' block and 'fc' layer     |     82.75%      |
|    Train 'layer4' block and 'fc' layer from scratch    |     -      |
|    Pretrained, Fine-tune Entire Model     |     90.32%      |
|    Train the Entire Model from scratch    |     87.55%      |

## Part 2: Object Detection
In this part, we implement the loss function of YOLO (eq. 3 in [this paper](https://pjreddie.com/media/files/papers/yolo_1.pdf)) and train an object detector on the PASCAL VOC 2007 dataset. 

### Backbone Network
The design of the backbone network is inspired by [DetNet](https://arxiv.org/abs/1804.06215). Specifically, the first four blocks of the backbone network is identical to a ResNet-50, enabling us to leverage the Imagenet pre-trained weight as initialization. We added an extra block while maintaining the spatial resolution, facilitating the model to capture smaller objects and more precise bounding boxes for large objects. We note that high-level features from deeper layers also benefit the classification (To detect an object, we predict its bounding box and class).

Please refer the YOLO loss in `yolo_loss.py` and the backbone network in `src/resnet_yolo.py` for details. 

## Result

<img height="250" width="375" alt="" src="output/output.gif" />