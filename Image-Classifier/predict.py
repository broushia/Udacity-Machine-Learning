# This is in a folder, use this as a shortcut
# python /home/workspace/aipnd-project/predict.py

# Test image and path
# python /home/workspace/aipnd-project/predict.py aipnd-project/flowers/test/63/image_05878.jpg aipnd-project/checkpoint.pth
import os
import sys

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

from functions import load_images, build_model, predict_input_args, load_model, predict

def main():
    #print("hello world") for luck
    
    in_arg = predict_input_args()
        
    print("   image =", in_arg.image_dir,"\n   model checkpoint =",in_arg.load_dir,
          "\n   top k =",in_arg.top_k, "\n   device =", in_arg.device, "\n   json =", in_arg.json)
    
    model, optimizer = load_model(in_arg.load_dir,in_arg.device)
         
    probs, classes ,labels = predict(in_arg.image_dir, model,in_arg.json, in_arg.top_k, in_arg.device)
    
    results = dict(zip(labels,probs))
    print("-"*40)
    for x in results:
        print("   {:20s}   {:.2f}%".format(x.title(),results[x]*100)) 

if __name__ == "__main__":
    main()    