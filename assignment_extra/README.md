# Extra Credit Assignment: Adversarial Attacks
## Attack methods
We attacked pre-trained Imagenet models provided by PyTorch, namely ResNet-18, ResNet-34, ResNet-50, ResNet-101, Inception V3, ViT_B_16, and VGG19 with Batch Normalization. We only attacked images in Imagenette. We implemented the following 3 attacks:
- Fast Gradient Sign (FGSM)
- Iterative Gradient Sign
- Least Likely Class Methods

### Results


## Robustness to non-adversarial distribution change
We applied "real world image degradations" using ImageNet-C perturbations on ImageNette and compare accuracy with original.

### Results
