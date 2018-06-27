import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, pdb, time, copy
from PIL import Image
import net
import trainer

if __name__ == "__main__":
    model = net.Net()
    model = model = nn.DataParallel(model, [0, 1])
    model = model.cuda()
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 200

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    input_scale_size = 32
    model_save_name = 'net.ckpt'
    data_dir = os.path.join('./data')
    batch_size = 64

    trainer = trainer.Train(mean=mean, std=std, input_scale_size=input_scale_size, data_dir=data_dir, model_save_name=model_save_name, batch_size=batch_size)

    pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': model.parameters()}], lr=LEARNING_RATE)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    trainer.train_model(model, criterion, optimizer, exp_lr_scheduler, NUM_EPOCHS)