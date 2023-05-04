import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet50, resnet34, resnet101, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models import vit_b_16 ,ViT_B_16_Weights


from pathlib import Path
import matplotlib.pyplot as plt

# imagenette dataset
# from fastai document

# transforms used by PyTorch pretrained resnet, https://pytorch.org/hub/pytorch_vision_resnet/
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        
def build_dataset(root: str, transform=None) -> Dataset:
    return ImageFolder(root=root, transform=transform()) if transform is not None \
           else ImageFolder(root=root, transform=trans)

def prepare_model(type: str, from_scratch: bool=False):
    
    pretrained = {
        'resnet18'  :  ResNet18_Weights.DEFAULT,
        'resnet34'  :  ResNet34_Weights.DEFAULT,
        'resnet50'  :  ResNet50_Weights.DEFAULT,
        'resnet101' :  ResNet101_Weights.DEFAULT,
        'inception' :  Inception_V3_Weights.DEFAULT,
        'vit'       :  ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        'vgg'       :  VGG19_BN_Weights.DEFAULT
    }

    weight = None
    if not from_scratch:
        weight = pretrained[type]

    if type == 'resnet18':
        return resnet18(weights=weight), ResNet18_Weights.IMAGENET1K_V1.transforms
    elif type == 'resnet34':
        return resnet34(weights=weight), ResNet34_Weights.IMAGENET1K_V1.transforms
    elif type == 'resnet50':
        return resnet50(weights=weight), ResNet50_Weights.IMAGENET1K_V2.transforms # for resnet50 and resnet101, V2 is default
    elif type == 'resnet101':
        return resnet101(weights=weight), ResNet101_Weights.IMAGENET1K_V2.transforms
    elif type == 'inception':
        return inception_v3(weights=weight), Inception_V3_Weights.IMAGENET1K_V1.transforms
    elif type == 'vit':
        return vit_b_16(weights=weight), ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms
    elif type == 'vgg':
        return vgg19_bn(weights=weight), VGG19_BN_Weights.IMAGENET1K_V1.transforms
    else:
        print('No such method being implemented')
        raise NotImplementedError

def prepare_loss(type: str):
    if type == 'nll':
        return nn.NLLLoss()
    elif type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        print('No such method being implemented')
        raise NotImplementedError

def plot_result(root, method, results, eps, name): 
    
    path = Path(root) / f'{method}_{name}'
    path.mkdir(parents=True, exist_ok=True)
    
    if method == 'all':
        for idx, result in enumerate(results):
            methods = ['FGSM', 'iter', 'least']
            markers = {
                'FGSM': '1',
                'iter': 'v',
                'least': '8'
            }
            lines = {
                'FGSM': 'dotted',
                'iter': 'dashed',
                'least': 'dashdot'
            }
            colors = {
                'FGSM': 'r',
                'iter': 'b',
                'least': 'g'
            }
            plt.xlabel('epsilon (pixels in range [0, 255])')
            plt.ylabel('Accuracy')
            plt.grid('True')
            for i in range(len(result)):
                plt.plot(eps, result[i], marker=markers[methods[i]], color=colors[methods[i]], label=methods[i], linestyle=lines[methods[i]])
            plt.legend()
            if idx == 0:
                plt.savefig(path / 'top_result.png', dpi=400)
            else:
                plt.savefig(path / 'top5_result.png', dpi=400)
            plt.close()
    else:
        plt.xlabel('epsilon (pixels in range [0, 255])')
        plt.ylabel('Top-1 Accuracy')
        plt.grid('True')
        plt.plot(eps, results[0], marker='.', label=method)
        plt.legend()
        plt.savefig(path / 'top_result.png', dpi=400)
        plt.close()

        plt.xlabel('epsilon (pixels in range [0, 255])')
        plt.ylabel('Top-5 Accuracy')
        plt.grid('True')
        plt.plot(eps, results[1], marker='.', label=method)
        plt.legend()
        plt.savefig(path / 'top5_result.png', dpi=400)
