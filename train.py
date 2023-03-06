#!/bin/python

import shutil
import torch
import os
import time
from module import *
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")

class MyTransform:
    def __init__(self):
        pass

    def __call__(self, img):
        img = tfs.functional.rotate(img, -90)
        img = tfs.functional.hflip(img)
        img = img.map_(img, lambda a, b: 1.0 - a)
        return img

trainset = EMNIST(root = "",
                split = "digits",
                train = True,
                download = True,
                transform = tfs.Compose([tfs.PILToTensor(),
                        tfs.Resize(29),
                        tfs.ConvertImageDtype(torch.float32),
                        MyTransform()
                        ]))
testset = EMNIST(root = "",
                split = "digits",
                train = False,
                download = True,
                transform = tfs.Compose([tfs.PILToTensor(),
                        tfs.Resize(29),
                        tfs.ConvertImageDtype(torch.float32),
                        MyTransform()
                        ]))

trainset_size = len(trainset)
testset_size = len(testset)

trainset_dataloader = DataLoader(trainset, batch_size = 1, shuffle = True);
testset_dataloader = DataLoader(testset, batch_size = 1, shuffle = True);

my_module = MyModule()
my_module.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(my_module.parameters(), lr = learning_rate)

log_dir = "log"
shutil.rmtree(log_dir, ignore_errors=True)
log = SummaryWriter(log_dir)

#log.add_graph(my_module, torch.ones((1, 1, 29, 29)))

print("Let's train")

step = 0
epoch = 10

for i in range(epoch):

    my_module.train()

    start = time.time()
    for data in trainset_dataloader:
        imgs, targets = data
        imgs, targets = (imgs.to(device), targets.to(device))
        output = my_module(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.add_scalar("loss", loss, step)
        log.add_images("imgs", imgs, step)

        step += 1
        if step % 100 == 0:
            end = time.time()
            print("training step {}, time {}".format(step, end - start))
            start = time.time()

    my_module.eval()

    torch.save(my_module, "my_module_{}.pth".format(i))

    total_loss_testset = 0
    total_right_predict = 0
    with torch.no_grad():
        for data in testset_dataloader:
            imgs, targets = data
            imgs, targets = (imgs.to(device), targets.to(device))
            output = my_module(imgs)
            total_loss_testset += loss_fn(output, targets).item()
            total_right_predict += (torch.argmax(output, dim=1) == targets).sum()

    accuracy_testset = total_right_predict / testset_size

    log.add_scalar("total_loss_testset", total_loss_testset, i)
    log.add_scalar("accuracy_testset", accuracy_testset, i)

    print("epoch {}: accuracy_testset {}, total_loss_testset {}".format(i,
          accuracy_testset, total_loss_testset))

log.close()
