from PIL import Image
from pathlib import Path

target = [Path('imagenette2'), Path('imagenette_c'), ]
head = Path('combine')

for idx, tar in enumerate(target):
    for i in tar.glob('**/*JPEG'):
        out = Path('combine')
        for j in i.parts[1:-1]:
            out /= j
        out /= f'{i.stem}_from_{idx}{i.suffix}'
        Path(out.parent).mkdir(parents=True,exist_ok=True)
        img = Image.open(i)
        img.save(out)
