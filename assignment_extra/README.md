# Extra Credit Assignment: Adversarial Attacks
## Attack methods
We attacked pre-trained Imagenet models provided by PyTorch, namely ResNet-18, ResNet-34, ResNet-50, ResNet-101, Inception V3, ViT_B_16, and VGG19 with Batch Normalization. We only attacked images in Imagenette. We implemented the following 3 attacks:
- Fast Gradient Sign (FGSM)
...
- Iterative Gradient Sign
...
- Least Likely Class Methods
...

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


## Robustness to non-adversarial distribution change
We applied "real world image degradations" using ImageNet-C perturbations on ImageNette and compare accuracy with original.

### Results
