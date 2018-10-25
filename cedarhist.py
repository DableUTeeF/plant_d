"""
/media/palm/Unimportant/pdr2018/typesep_train/Cedar
"""

from __future__ import print_function
import os
import warnings

warnings.simplefilter("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as f
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
from utils import progress_bar, format_time
import models
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import autoaugment as aug
import time
from imagemove2 import getlookup
from matplotlib import pyplot as plt

# todo: 0). Try https://github.com/DeepVoltaire/AutoAugment
# todo: 1). DenseNet201 with Conv1D as output. Dense6
# todo: 2). NasNetMobile with droppath. Nas1
# todo: 3). If 2nd goes well, implement droppath to another models.


def getmodel(cls=61, resume_path=None):
    model_conv = models.densenet.densenet201(pretrained=False, num_classes=61)
    if resume_path:
        checkpoint = torch.load(resume_path)
        model_conv.load_state_dict(checkpoint['net'])
    num_ftrs = model_conv.classifier.in_features
    # model_conv.classifier = nn.Conv1d(num_ftrs, cls, kernel_size=(1, 1))
    model_conv.classifier = nn.Linear(num_ftrs, cls)

    return model_conv


class Conv1dClassifier(nn.Module):
    def __init__(self, in_ch, cls):
        super().__init__()
        self.classifier = nn.Conv1d(in_ch, cls, kernel_size=(1, 1))

    def forward(self, x):
        # x = f.alpha_dropout()
        x = self.classifier(x)
        return x


class FcClassifier(nn.Module):
    def __init__(self, num_ftrs, cls):
        super().__init__()
        self.fc1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, cls)

    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        x = f.dropout(x, 0.5)
        x = self.fc2(x)
        return x


class ReplicatePad:
    def __init__(self, size):
        assert isinstance(size, int), 'size should be an int'
        self.size = size

    def __call__(self, img):
        width, height = img.size
        if width > height:
            img = img.resize((self.size, int(height / width * self.size)))
        else:
            img = img.resize((int(width / height * self.size), self.size))

        if width > height:
            return F.pad(img, (0, 0, self.size - img.size[0], 0), padding_mode='constant')
        else:
            return F.pad(img, (0, 0, 0, self.size - img.size[0]), padding_mode='constant')


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


# noinspection PyUnresolvedReferences
class CustomSGD(torch.optim.SGD):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                nomial = np.random.binomial(1, 0.9, d_p.size())
                nomial = torch.FloatTensor(nomial)
                d_p = torch.addcmul(torch.zeros(a.size(), dtype=torch.FloatTensor), d_p, nomial)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


if __name__ == '__main__':

    lookup = getlookup()
    args = DotDict({
        'batch_size': 32,
        'batch_mul': 4,
        'val_batch_size': 10,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': [60],
        'try_no': '1_densecedar',
        'imsize': [224],
        'imsize_l': [256],
        # 'traindir': '/root/palm/DATA/plant/typesep_train/Cedar',
        'valdir': '/media/palm/Unimportant/pdr2018/typesep_train/Cedar',
        'workers': 4,
        'resume': False,
    })

    # try:
    #     print(f'loading log: log/try_{args.try_no}.json')
    #     log = eval(open(f'log/try_{args.try_no}.json', 'r').read())
    # except FileNotFoundError:
    log = {'acc': [], 'loss': [], 'val_acc': []}
    # print(f'Log {args.try_no} not found')
    zz = 0
    i = 0
    best_acc = 0
    best_no = 0
    start_epoch = 1

    model = getmodel(cls=1).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=False, )
    scheduler = MultiStepLR(optimizer, [20, 40])
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.BCELoss().cuda()
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.valdir),
                             transforms.Compose([
                                 transforms.Resize(args.imsize_l[i]),
                                 transforms.CenterCrop(args.imsize[i]),
                                 transforms.ToTensor(),

                                 normalize,
                             ])),
        batch_size=1,
        num_workers=args.workers,
        pin_memory=False)

    cudnn.benchmark = args.batch_size > 1
    args['resume'] = f'./checkpoint/try_{args.try_no}best.t7'
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['acc']
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    model.eval()
    # test_loss = 0
    correct = 0
    total = 0
    hist = [[], []]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            # targets = targets.view(10).type(torch.cuda.FloatTensor)
            outputs = model(inputs.view(-1, 3, args.imsize[i], args.imsize[i]))
            # loss = criterion(outputs, targets)
            # test_loss += loss.cpu().item()
            outputs = f.sigmoid(outputs)
            hist[targets].append(int(outputs.cpu().detach().numpy()))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    plt.plot(hist[0])
    plt.plot(hist[1])
    plt.show()
