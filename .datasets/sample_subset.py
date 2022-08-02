"""
Creates a random subset of the ImageNet validation-set with 10k samples.
"""

import random
import os
import shutil

random.seed(123)

path = 'imagenet/val'
copy_path = "imagenet10k/val"

if not os.path.exists("imagenet10k"):
    os.mkdir("imagenet10k")
if not os.path.exists(copy_path):
    os.mkdir(copy_path)

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    images = os.listdir(folder_path)
    subset = random.sample(images, 10)
    dst = os.path.join(os.getcwd(), copy_path, folder)
    if not os.path.exists(dst):
        os.mkdir(dst)
    for image in subset:
        src = os.path.join(folder_path, image)
        shutil.copy(src, dst)
