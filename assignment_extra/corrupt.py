from imagenet_c import corrupt
from pathlib import Path
import numpy as np
import cv2 as cv


path = Path('imagenette2/val')
out_path = Path('imagenette_c/val')
out_path.mkdir(exist_ok=True, parents=True)

for img_path in path.glob('**/*JPEG'):
    # img = cv.imread(img_path)
    print(img_path)
    # severity, type_ = np.random.randint(0, 5), np.random.randint(0, 18)
    # img_c = corrupt(x=img, severity=severity, corruption_number=type_)
    # cv.imwrite(out_path)
