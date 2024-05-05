import os
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.resnet_yolo import resnet50
from src.predict import predict_image
from src.config import VOC_CLASSES, COLORS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_network_path = 'checkpoints/detector_epoch_70.pth' 
pretrained = True

if load_network_path is not None:
    print('Loading saved network from {}'.format(load_network_path))
    net = resnet50().to(device)
    net = torch.compile(net)
    net.load_state_dict(torch.load(load_network_path))
else:
    print('Load pre-trained model')
    net = resnet50(pretrained=pretrained).to(device)


net.eval()

# select random image from test set
inference_file_root = 'inference/pic/'
inference_pic = sorted(os.listdir(inference_file_root))
for img_name in inference_pic:
    print('processing ', img_name)
    image = cv2.imread(os.path.join(inference_file_root, img_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print('predicting...', img_name)
    result = predict_image(net, img_name, root_img_directory=inference_file_root)
    for left_up, right_bottom, class_name, _, prob in result:
        color = COLORS[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                    color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    plt.figure(figsize = (15,15))
    plt.imsave('inference/result/'+img_name, image)
