import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from config import *

def find_max_lives(env):
    env.reset()
    _, _, _, _, info = env.step(0)
    return info['lives']

def check_live(life, cur_life):
    if life > cur_life:
        return True
    else:
        return False

def get_frame(X):
    x = np.uint8(resize(rgb2gray(X[0]), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x

def get_init_state(history, s, history_size):
    for i in range(history_size):
        history[i, :, :] = get_frame(s)
