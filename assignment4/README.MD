# Assignment 4: Cat Face Generation with GANs

In this assigment, we play with various GAN techniques. We implement the [orignial GAN loss](https://arxiv.org/pdf/1406.2661.pdf), the [least squares GAN loss (LS-GAN)](https://arxiv.org/abs/1611.04076), WGAN loss, and GAN inversion. We elaborate the details of our implementation in the following bullet points.

- GAN Loss and LS-GAN loss
In `gan/losses.py`, we implement the two losses for training GAN. To train a GAN with these losses on the cat, refer to `MP4.ipynb`. Note that you should first run `sh download_cat.sh` to download the cat dataset. Use `Extra_Credit_train_abstract.ipynb` to train GANs on [the abstract art gallery dataset](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery).

- WGAN Loss
In `gan/losses.py`, we implement the WGAN loss and function to compute the gradient penalty. The WGAN loss enables a more stable training trajectory while mitigating the mode collapse problem. We recommend you to refer to [this blog](https://lilianweng.github.io/posts/2017-08-20-gan/), as the author explained the theory of WGAN thoroghly along with mathematical derivation and summarized advantages of WGAN over original GANs.

- GAN Inversion
We implement the GAN inversion technique proposed in [this paper](https://arxiv.org/abs/2004.00049). Specifically, a domain-guided encoder is trained to encode an image to a latent code in the latent space of the generator. Given a latent code $z$, a generator $G$, the encoder $E$ is trained to encode $G(E(z))$ to $z$, i.e., the encoder maps an image to its associating latent code. The author also proposed to train the encoder with a discriminator, allowing the encoder to better fit the probabilistic distribution. Please refer to eq. 2 and 3 in the paper for more details.

## Result
Organizing...