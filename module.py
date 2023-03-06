#!/bin/python

import torch.nn as nn
import torch

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # from https://arxiv.org/abs/1202.2745
        self.module = nn.Sequential(nn.Conv2d(1, 20, 4),
				    nn.MaxPool2d(2),
				    nn.Conv2d(20, 40, 5),
				    nn.MaxPool2d(3),
				    nn.Flatten(),
				    nn.Linear(40 * 3 * 3, 150),
				    nn.Linear(150, 10))

    def forward(self, x):
    	return self.module(x)


if __name__ == '__main__':
    print("Test the module")
    my_module = MyModule()
    my_module.eval()
    input = torch.ones((1, 1, 29, 29))
    output = my_module(input)
    print(output)
else:
    print("Import module!")

