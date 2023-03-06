#!/bin/python

import shutil
import torch
import torchvision
from module import *
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torch.utils.tensorboard import SummaryWriter

my_module = torch.load("my_module_0.pth")

log_dir = "log_for_test"
shutil.rmtree(log_dir, ignore_errors=True)
log = SummaryWriter(log_dir)

img = torchvision.io.read_image("img_fortest/2.jpg", mode = torchvision.io.image.ImageReadMode.GRAY)
img = tfs.functional.resize(img, (29, 29))
img = img.reshape((-1, 1, 29, 29))
img = tfs.functional.autocontrast(img)
img = tfs.functional.convert_image_dtype(img, dtype = torch.float32)
log.add_images("test_imgs", img, 1)
log.close()
print(my_module(img))
print(my_module(img).argmax().item())
