# This is in a folder, use this as a shortcut
# python /home/workspace/aipnd-project/train.py aipnd-project

import numpy as np
import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb

import argparse

from functions import load_images, build_model, train_input_args, train_model, save_model

def main():
    #print("hello world") for luck
    
    in_arg = train_input_args()
    
    print("   data directory =",in_arg.data_dir, "\n   save directory =", in_arg.save_dir, "\n   model architecture =", in_arg.arch,
          "\n   hidden units =", in_arg.hidden_units, "\n   learning rate =", in_arg.learning_rate,
          "\n   epochs =", in_arg.epochs, "\n   device =", in_arg.device)
    
    image_data, data_loader = load_images()
    
    model, optimizer, criterion = build_model(in_arg.arch , in_arg.hidden_units, in_arg.learning_rate)
    
    model = train_model(model,data_loader, in_arg.epochs, criterion, optimizer, in_arg.device)
    
    save_model(model, optimizer, in_arg.arch, in_arg.data_dir, in_arg.save_dir,image_data)
    
    print("-"*40)
      

if __name__ == "__main__":
    main()    
