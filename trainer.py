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
import imgaug as ia
from imgaug import augmenters as iaa

iaa_seq = iaa.Sequential([
            iaa.PerspectiveTransform(scale=0.075),
            iaa.Multiply((0.8, 1.2)),
            iaa.Affine(
                scale={"x":(1, 1.1), "y":(1, 1.1)}
            ),
            iaa.SaltAndPepper((0.05)),
            iaa.Add((-20, 20)),
            iaa.GaussianBlur((0, 0.50))
        ])

def convert_train(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if np.random.randint(2) == 0:    
        img = iaa_seq.augment_image(np.array(img))
        img = Image.fromarray(img)
    # img = img.resize((self.IMAGE_SCALE_SIZE, self.IMAGE_SCALE_SIZE))
    return img

class ImageAugmentaion(object):
    def __call__(self, pic):
        return convert_train(pic)

    def __repr__(self):
        return self.__class__.__name__+'()'

transforms.ImageAugmentaion = ImageAugmentaion

class Train():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], input_scale_size=224, data_dir='/data/synthesized/model', model_save_name='model.ckpt', batch_size=64):
        self.DATA_DIR = data_dir
        self.MODEL_SAVE_NAME = model_save_name
        self.BATCH_SIZE = batch_size
        self.IMAGE_SCALE_SIZE = input_scale_size
        self.MEAN = mean
        self.STD = std
        self.data_transforms = {
                                    'train': transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ImageAugmentaion(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)
                                    ]),
                                    'test': transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)
                                    ])
                                }
        self.image_datasets = {x: torchvision.datasets.CIFAR10(root='./data', train= (x=='train'),
                                        download=True, transform=self.data_transforms[x]) for x in ['train', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'test']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'test']}
        self.class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.iaa_seq = iaa.Sequential([
            iaa.PerspectiveTransform(scale=0.050),
            iaa.Multiply((0.8, 1.2)),
            iaa.Affine(
                scale={"x":(1, 1.1), "y":(1, 1.1)}
            ),
            iaa.SaltAndPepper((0.05)),
            iaa.Add((-20, 20)),
            iaa.GaussianBlur((0, 0.50))
        ])

    def convert_train(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if np.random.randint(2) == 0:    
            img = self.iaa_seq.augment_image(np.array(img))
            img = Image.fromarray(img)
        # img = img.resize((self.IMAGE_SCALE_SIZE, self.IMAGE_SCALE_SIZE))
        return img

    def convert(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img = img.resize((self.IMAGE_SCALE_SIZE, self.IMAGE_SCALE_SIZE))
        return img

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'test']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = running_corrects_0 = running_corrects_1 = 0.0
                num_label_0 = num_label_1 = 0

                for inputs, labels in tqdm(self.dataloaders[phase]):
                    # pdb.set_trace()
                    inputs = inputs.float()
                    inputs, labels = inputs.cuda(), labels.cuda()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).view(-1, len(self.class_names))
                        _, preds = torch.max(outputs, 1)
                        try:
                            loss = criterion(outputs.squeeze(-1), labels)
                        except:
                            pdb.set_trace()
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)

                    running_corrects_0 += torch.sum((preds == labels.data) * (preds == 0))
                    num_label_0 += torch.sum(labels.data == 0)
                    num_label_1 += torch.sum(labels.data == 1)

                    running_corrects_1 += torch.sum((preds == labels.data) * (preds == 1))
                    running_corrects += torch.sum(preds == labels.data)

                try:
                    epoch_loss = running_loss/self.dataset_sizes[phase]
                    epoch_acc =  running_corrects.item()/self.dataset_sizes[phase]
                    epoch_acc_0 =  running_corrects_0.item()/num_label_0.item()
                    epoch_acc_1 =  running_corrects_1.item()/num_label_1.item()
                except:
                    pdb.set_trace()
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, self.class_names[0], epoch_acc_0, self.class_names[1], epoch_acc_1))

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join('ckpt', self.MODEL_SAVE_NAME))
                    print("model saved")
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)

        return model

    def load_model(self, model, ckpt_name):
        model.load_state_dict(torch.load(os.path.join('ckpt', ckpt_name)))

if __name__ == '__main__':
    pdb.set_trace()