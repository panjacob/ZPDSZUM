import os
from pathlib import Path

from PIL import Image, ImageOps


def crop_center(pil_img, crop_width=256, crop_height=256):
    img_width, img_height = pil_img.size
    if img_width == img_height:
        return pil_img

    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


path = 'ZPD/new'
new_path = path + '_256x256'
Path(new_path).mkdir(parents=True, exist_ok=True)
files = os.listdir(path)
files_len = len(files)
start = 584
for i, file in enumerate(files):
    print(f"{i + 1}/{files_len}")
    image = Image.open(os.path.join(path, file))
    image = ImageOps.exif_transpose(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = crop_center(image)
    new_image = image.resize((256, 256))

    new_image.save(os.path.join(new_path, f"{i + start}.jpg"))
