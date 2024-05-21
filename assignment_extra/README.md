# Extra Credit Assignment: Adversarial Attacks
## Attack methods
In this assignment, we attack pre-trained Imagenet models provided by PyTorch, namely ResNet-18, ResNet-34, ResNet-50, ResNet-101, Inception V3, ViT_B_16, and VGG19 with Batch Normalization with the following 3 attacks in this [paper](https://arxiv.org/pdf/1607.02533):
- Fast Gradient Sign (FGSM) 

The image itself is treated as trainable parameters. The image is fed to the targeted neural network and the gradient of the image is calculated by back propagation. The image is then perturbed in the direction of the signed gradient vector to increase the loss, i.e., increasing the chance that the model misclassify this image.
- Iterative Gradient Sign

Unlike, the FGSM method which only takes a step to perturb the image, the Iterative Gradient Sign method takes multiple gradient steps to perturbed the image. However, after each iteration, the perturbed image is clipped to ensure that it is in the $\epsilon$-neighbourhood of the original image.  
- Iterative Least Likely Class Method

Previous methods perturbed the image in the direction that increases the loss function. For the Iterative Least Likely Class Method, we perturbed the image, letting the model classifies it to the class that is significantly dissimilar (i.e., the class corresponding to the least element of the output logit) to the original class. 

Note that we only attack images in Imagenette. 

## Perform an attack
#### Perparation
Store all images that you want to generate their adversarials in a single folder.

#### Usage
Execute `main.py` to perform an attack. `main.py` reads an config file which specifies the attack method, path to the candidate images, pre-trained model to attack, path to store adversarial images, etc. If you want to generate adversarial images corresponded to all three methods with a single execution, set `method='all'`. Note the config files we used are stored in `config` and to use them, __you should download imagenette__ [here](https://github.com/fastai/imagenette?tab=readme-ov-file). 

For example, to generate adversarial images for ResNet-18, use the following command:
```console
python main.py --config configs/resnet18.yaml
```

### Results
Organizing...

## Robustness to non-adversarial distribution change
We applied "real world image degradations" using ImageNet-C perturbations on ImageNette and compare accuracy with original. To create a corrupted ImageNette dataset, execute `corrupt.py`:
```console
python corrupt.py 
```
The script creates a corrupted ImageNette dataset by randomly applying one of ImageNet-C perturbations on an ImageNette image.

The training script is `train_imagenette_c.py`. To run the code, execute:
```console
python train_imagenette_c.py --config config/imagnette_c.yaml
```

### Results
We test how ImageNet-C perturbations affect performance of a ResNet-34 network. The result is as follows:
| Setting | Test Acc. |
|---------|-----------|
|    Imagenet Pretrained, Test on ImageNette    |      84.62%     |
|    Imagenet Pretrained, Test on corrupted ImageNette     |     62.01%      |
|    Train on corrupted ImageNette, Test on ImageNette      |     91.96%      |
|    Train on corrupted ImageNette, Test on corrupted ImageNette     |     90.82%      |
|    Train on corrupted ImageNette and ImageNette, Test on corrupted ImageNette and ImageNette     |     91.98%      |
|    Pretrained ResNet finetune on corrupted ImageNette, Test on corrupted ImageNette     |     78.39%      |
|    Pretrained ResNet finetune on corrupted ImageNette, Test on corrupted ImageNette and ImageNette     |     82.57%      |