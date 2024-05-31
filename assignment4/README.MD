# Assignment 4: Cat Face Generation with GANs

In this assignment, we play with various GAN techniques. We implement the [original GAN loss](https://arxiv.org/pdf/1406.2661.pdf), the [least squares GAN loss (LS-GAN)](https://arxiv.org/abs/1611.04076), WGAN loss, and GAN inversion. We elaborate on the details of our implementation in the following sections.

## GAN Loss, LS-GAN loss, and Spectrally Normalized GAN (SNGAN)
In `gan/losses.py`, we implement the two losses for training GAN. To train a GAN with these losses on the cat, refer to `MP4.ipynb`. Use `Extra_Credit_train_abstract.ipynb` to train GANs on [the abstract art gallery dataset](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery). To activate the spectral normalization in the discriminator, set `spectral_norm=True` when creating a discriminator.

### Usage 
Download the cat face dataset:
```console
sh download_cat.sh
```
Then feel free to play around with `MP4.ipynb`.

Download the the abstract art gallery dataset via the [link](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery) then you are ok to run `Extra_Credit_train_abstract.ipynb`.

## WGAN Loss

In `gan/losses.py`, we implement the WGAN loss and function to compute the gradient penalty. The WGAN loss enables a more stable training trajectory while mitigating the mode collapse problem. We recommend you refer to [this blog](https://lilianweng.github.io/posts/2017-08-20-gan/), as the author explained the theory of WGAN thoroughly along with mathematical derivation and summarized advantages of WGAN over original GANs.

### Usage
Same as the prior section.

## GAN Inversion

We implement the GAN inversion technique proposed in [this paper](https://arxiv.org/abs/2004.00049). Specifically, a domain-guided encoder is introduced to facilitate the inversion process. The encoder is trained to encode an image to a latent code in the latent space of the generator. Given a latent code $z$, a generator $G$, the encoder $E$ is trained to encode $G(E(z))$ to $z$, i.e., the encoder maps an image to its associating latent code. The author also proposed to train the encoder with a discriminator, allowing the encoder to better fit the probabilistic distribution. Please refer to eq. 2 and 3 in the paper for more details. 

We note that the domain-guided encoder itself essentially inverts the image to a latent code. However, to enhance the quality of the inversion, the latent code should be refined via optimization. Please check eq. 4 in the paper for details.

### Usage
To train the domain-guided encoder, run
```console
python Extra_Credit_GAN_domain_encoder.py
```

To invert an image to a latent code, run
```console
python Extra_Credit_inverter.py --config config/naive.yaml
```
In the config file, you can specify the image to be inverted, the generator to be used, the path to save the image generated via the inverted code, and so on.

## Result
Organizing...