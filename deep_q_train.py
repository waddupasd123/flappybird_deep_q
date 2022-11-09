import numpy as np
import torch
import torch.nn as nn
from conv_model import FlappyConv

def train():
    model = FlappyConv()

    if torch.cuda.is_available():  
        dev = "cuda:0"
        model.cuda()
        image = image.cuda() 
    else:  
        dev = "cpu"  

