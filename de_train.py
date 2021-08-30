# Author: IconBall
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import sys
from PIL import Image
import warnings
from tqdm import tqdm
import time
from torchvision import datasets, transforms
import torchvision
import copy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
from utils.generate_dataset import GenerateDataset

models = ['resnet50', 'mobile', 'resnet34']

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='Model',
                    required=True, choices=models)
# Model tag
parser.add_argument('-to', '--tag_old',
                    help='tag of old model', type=str, default='')
parser.add_argument('-tn', '--tag_new',
                    help='tag of new model', type=str, default='')

# image path
parser.add_argument('-d', '--dir', help='input directory',
                    type=str, default='./barknet_scale/')

parser.add_argument('-device', '--device', help='input directory',
                    type=str, default='cuda:0')
# Split seed
parser.add_argument('-seed', '--seed',
                    help='data split seed', type=int, default=1024)

# paremeter path
parser.add_argument('-pd', '--paremeter_dir', help='paremeter direction',
                    type=str, default='./checkpoint/')

# test model
parser.add_argument(
    '--test_model', help='test model option', action='store_true')

# Non local mean:
parser.add_argument('-de', '--is_denoise',
                    help='add denoise layer', action='store_true')
parser.add_argument('-dn', '--denosie_num',
                    help='the number of denoise block defalut:4', type=int, default=4)

# Cot Training
parser.add_argument('-cot', '--is_completment_train',
                    help='Guided Complement Entropy Training', action='store_true')
parser.add_argument('-cota', '--cot_alpha',
                    help='alpha value for Guided Complement Entropy Training', type=float, default=0.2)

# Learning Paremeter
parser.add_argument('-bs', '--batch_size',
                    help='mini batch size', type=int, default=32)
parser.add_argument('-i', '--input_size',
                    help='image input size', type=int, default=224)
parser.add_argument('-lr', '--learning_rate',
                    help='inital learning rate', type=float, default=1e-4)
parser.add_argument('-nc', '--num_classes',
                    help='classes number', type=float, default=10)
parser.add_argument('-ne', '--num_epochs',
                    help='epochs', type=float, default=40)
parser.add_argument('-es', '--early_stop_round',
                    help='early stop round', type=int, default=20)
parser.add_argument('-ls', '--label_smoothing',
                    help='label smoothing', type=float, default=0.1)

parser.add_argument('-f', '--fold',
                    help='num of fold', type=int, default=0)

args = parser.parse_args()
args.tag_old = '_' + args.tag_old if len(args.tag_old) > 0 else args.tag_old
args.tag_new = '_' + args.tag_new if len(args.tag_new) > 0 else args.tag_new

warnings.simplefilter("ignore", UserWarning)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


class GetModel():
    def __init__(self):
        self.models = ['resnet50', 'mobile', 'resnet34']

    def __call__(self, model_name, input_size, is_denoise=True, denosie_num=None):
        if model_name in self.models:
            if model_name == 'resnet50':
                model = torchvision.models.resnet50(pretrained=True)
                model.fc = nn.Linear(2048, 20)

            if model_name == 'resnet34':
                model = torchvision.models.resnet34(pretrained=True)
                model.fc = nn.Linear(512, 20)

            if model_name == "mobile":
                model = torchvision.models.mobilenet_v2(pretrained=True)
                model.classifier[1] = nn.Linear(1280, 20)
            setattr(model, 'input_size', (3, input_size, input_size))
            return model
        else:
            print('Current this model is not allowed')


def train_model(model, dataloaders, criterions, params_to_update, num_epochs=25, is_inception=False, best_acc=0,
                early_stop_round=5, lr=0.045, batch_update=1):
    for epoch in range(1, num_epochs + 1):
        if (epoch == 16) or (epoch == 33):
            lr = lr / 5
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_corrects = 0
            # Iterate over data.
            dataphase = tqdm(dataloaders[phase])

            for inputs, labels in dataphase:

                if phase == 'val':
                    dataphase.set_description(
                        f"[{model_name}][{epoch}] evaluating")
                else:
                    # dataphase.set_description(f"[{model_name}][{epoch}][{best_acc:.4f}][{last_epoch_acc:.4f}]")
                    dataphase.set_description(f"[{model_name}][{epoch}][{best_acc:.4f}]")
                if phase == 'train':
                    for criterion in criterions:
                        for batch_idx in range(batch_update):
                            t_input = inputs[(batch_idx) * args.batch_size // batch_update:(
                                batch_idx + 1) * args.batch_size // batch_update].to(
                                device)
                            t_label = labels[(batch_idx) * args.batch_size // batch_update:(
                                batch_idx + 1) * args.batch_size // batch_update].to(
                                device)
                            if len(t_input) == 0:
                                break
                            with torch.set_grad_enabled(True):
                                if is_inception and phase == 'train':
                                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                    # outputs, aux_outputs = model(inputs)
                                    # loss1 = criterion(outputs, labels)
                                    # loss2 = criterion(aux_outputs, labels)
                                    # loss = loss1 + 0.4*loss2
                                    pass
                                else:
                                    outputs = model(t_input)
                                    loss = criterion(
                                        outputs, t_label) / batch_update  # /len(criterions)
                                    loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                elif phase == 'val':
                    for batch_idx in range(batch_update):
                        t_input = inputs[(batch_idx) * args.batch_size // batch_update:(
                            batch_idx + 1) * args.batch_size // batch_update].to(
                            device)
                        t_label = labels[(batch_idx) * args.batch_size // batch_update:(
                            batch_idx + 1) * args.batch_size // batch_update].to(
                            device)
                        if len(t_input) == 0:
                            break
                        with torch.set_grad_enabled(False):
                            if is_inception and phase == 'train':
                                pass
                            else:
                                outputs = model(t_input)
                        _, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds == t_label.data)

            if phase == 'val':
                epoch_acc = running_corrects.item() / len(dataloaders[phase].dataset)
                if epoch_acc > best_acc:
                    early_stop_counter = 0
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(
                    ), f'{args.paremeter_dir}{model_name}_{input_size}{args.tag_new}.pth')
                else:
                    early_stop_counter += 1
                last_epoch_acc = epoch_acc
        
        #torch.save(model.state_dict(
        #), f'{args.paremeter_dir}{args.model}_{args.fold}_{epoch}.pth')
        
    return model


if __name__ == '__main__':
    model = GetModel()(model_name=args.model, input_size=args.input_size,
                       is_denoise=args.is_denoise, denosie_num=args.denosie_num)
    input_size = args.input_size
    model_name = args.model
    is_inception = False if model_name != 'InceptionV3' else True

    #try:
    # model.load_state_dict(torch.load(
    # f'{args.paremeter_dir}{args.model}_{args.input_size}{args.tag_old}.pth'))
    # print('load model successfully')
    # except:
    # print('load model fail, train from scratch')

    train_transform = torchvision.transforms.Compose([
       # torchvision.transforms.RandomCrop((input_size * 2, input_size * 2)),
        
        torchvision.transforms.RandomCrop(input_size,pad_if_needed=True),
        torchvision.transforms.RandomOrder([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomGrayscale(p=0.1)
        ]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x * 2 - 1),
    ])

    train = torchvision.datasets.ImageFolder(
        root=args.dir, transform=train_transform)
    
    dataset = GenerateDataset(args.dir)
    dataset.load_dataset(existing_dataset='None')
    
    train.samples = dataset.get_k_fold_format_dataset(args.fold,5)[0]

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,num_workers=8)
    dataloaders_dict = {'train': trainloader}
    device = torch.device(args.device)  # if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)
    model.train()
    params_to_update = model.parameters()

    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

    # if args.is_completment_train:
    # criterions = [GCEloss(alpha=args.cot_alpha), BCEWithLogitsLossAndSmooth(
    # reduction='mean', label_smoothing=args.label_smoothing)]
    # else:
    criterions = [nn.CrossEntropyLoss(reduction='mean')]
    # Train and evaluate
    model = train_model(model=model, dataloaders=dataloaders_dict, criterions=criterions,
                        params_to_update=params_to_update,
                        num_epochs=args.num_epochs, is_inception=is_inception, best_acc=0,
                        early_stop_round=args.early_stop_round, lr=args.learning_rate, batch_update=1)
    torch.save(model.state_dict(
    ), f'{args.paremeter_dir}{args.model}_{args.fold}_final.pth')
