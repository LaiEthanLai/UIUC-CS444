from imagenet_c import corrupt
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle

types = [
    'gaussian_noise', 
    'shot_noise', 
    'impulse_noise', 
    'defocus_blur',
    'glass_blur', 
    'motion_blur',
    'zoom_blur', 
    'snow',
    'frost', 
    'fog', 
    'brightness',
    'contrast', 
    'elastic_transform',
    'pixelate', 
    'jpeg_compression', 
    'speckle_noise', 
    'gaussian_blur',
    'spatter', 
    'saturate'
]

record = {k : {kk : 0 for kk in range(6)} for k in types}

path = Path('imagenette2/train')
out_path_head = Path('imagenette_c/train')
out_path_head.mkdir(exist_ok=True, parents=True)

with tqdm(path.glob('**/*JPEG')) as tq:
    for img_path in tq:
        
        severity, type_ = np.random.randint(0, 6), np.random.randint(0, 18)
        img = Image.open(img_path)
        while type_  in [8, 17]: # imagenet_c doesnt support adjusting brightness, saturation of gray imgs
            type_ = np.random.randint(0, 18)
        
        if img.mode != 'RGB': type_ = 0
        
        record[types[type_]][severity] += 1
        tq.set_description_str(f'corrupting {img_path.name} with severity: {severity}, type: {types[type_]}')
        
        out_path = out_path_head
        for folder in img_path.parts[2:]:
            out_path = out_path / folder
        tq.set_postfix_str(f'write to {out_path.parent}')
        out_path.parent.mkdir(exist_ok=True, parents=True)
        
        img = img.resize(size=(224, 224), resample=Image.Resampling.BILINEAR) 
        img = np.asarray(img)
        img_c = corrupt(x=img, corruption_number=type_)
        img_c = Image.fromarray(img_c)
        img_c.save(out_path)

print(record)
with open('record.pickle', 'rb') as f:
    pickle.dump(record, f)