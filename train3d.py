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


# todo: 1). Try Adam


def getmodel(cls=61):
    # model_conv = models.nasnet.nasnetmobile(cls)
    model_conv = models.densenet.densenet201(pretrained=True)
    num_ftrs = model_conv.classifier.in_features
    # model_conv.classifier = nn.Conv1d(num_ftrs, cls, kernel_size=(1, 1))
    model_conv.classifier = nn.Linear(num_ftrs, cls)
    return model_conv


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


if __name__ == '__main__':

    args = DotDict({
        'batch_size': 8,
        'batch_mul': 4,
        'val_batch_size': 10,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': [90],
        'try_no': '1_3ddensev2',
        'imsize': [224],
        'imsize_l': [256],
        'traindir': '/root/palm/DATA/plant/train/',
        'valdir': '/root/palm/DATA/plant/validate/',
        'workers': 16,
        'resume': False,
    })
    best_acc = 0
    best_no = 0
    start_epoch = 1
    try:
        print(f'loading log: log/try_{args.try_no}.json')
        log = eval(open(f'log/try_{args.try_no}.json', 'r').read())
    except FileNotFoundError:
        log = {'acc': [], 'loss': [], 'val_acc': []}
        print(f'Log {args.try_no} not found')
    # model = getmodel(61).cuda()
    model = models.resnet3D.DenseNet3DV2().cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # loss, acc, val_acc:
    # 128   0.001: [4.07, 7.33, ~8], 0.01: []
    # 32    0.001: [], 0.01: []
    optimizer = torch.optim.SGD(model.parameters(), 0.01,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=False, )
    # scheduler = ExponentialLR(optimizer, 0.97)
    scheduler = MultiStepLR(optimizer, [20, 40, 60])
    # scheduler2 = MultiStepLR(optimizer, [2], gamma=5)
    # platue = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.2)
    criterion = nn.CrossEntropyLoss().cuda()
    zz = 0
    for i in range(len(args.epochs)):
        train_dataset = datasets.ImageFolder(
            args.traindir,
            transforms.Compose([
                transforms.Resize(args.imsize_l[i]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),

                transforms.FiveCrop(args.imsize[i]),
                transforms.Lambda(
                    lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
                # transforms.ToTensor(),
                # normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.valdir, transforms.Compose([
                transforms.Resize(args.imsize_l[i]),
                # ReplicatePad(args.imsize_l[i]),
                # transforms.CenterCrop(args.imsize[i]),
                transforms.FiveCrop(args.imsize[i]),
                transforms.Lambda(
                    lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),

                # transforms.ToTensor(),
                # normalize,
            ])),
            batch_size=args.val_batch_size,
            num_workers=args.workers,
            pin_memory=False)

        # model = torch.nn.parallel.DistributedDataParallel(model).cuda()
        # model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = args.batch_size > 1
        if args.resume:
            if args.resume is True:
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


        def train(epoch):
            print('\nEpoch: %d/%d' % (epoch, args.epochs[i]))
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            optimizer.zero_grad()
            start_time = time.time()
            last_time = start_time
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, targets) / args.batch_mul
                loss.backward()
                train_loss += loss.item() * args.batch_mul
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                lfs = (batch_idx + 1) % args.batch_mul
                if lfs == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step_time = time.time() - last_time
                last_time = time.time()
                try:
                    _, term_width = os.popen('stty size', 'r').read().split()
                    print(f'\r{" "*(int(term_width))}', end='')
                except ValueError:
                    pass
                lss = f'{batch_idx}/{len(trainloader)} | ' + \
                      f'ETA: {format_time(step_time*(len(trainloader)-batch_idx))} - ' + \
                      f'loss: {train_loss/(batch_idx+1):.{3}} - ' + \
                      f'acc: {correct/total:.{5}}'
                print(f'\r{lss}', end='')

            # progress_bar(batch_idx, len(trainloader),
            #              f'Loss: {train_loss/(batch_idx+1):.{3}} | Acc: {100.*correct/total:.{3}}%')
            # else:
            #     print(f'sync {t2}, forw {t3}, loss {t4}, bck {t5}, oth {t6}', end='\r ')

            print(f'\r '
                  f'ToT: {format_time(time.time() - start_time)} - '
                  f'loss: {train_loss/(batch_idx+1):.{3}} - '
                  f'acc: {correct/total:.{5}}', end='')
            optimizer.step()
            # scheduler2.step()
            log['acc'].append(100. * correct / total)
            log['loss'].append(train_loss / (batch_idx + 1))


        def test(epoch):
            global best_acc, best_no
            model.eval()
            # test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')
                    outputs = model(inputs)
                    # loss = criterion(outputs, targets)

                    # test_loss += loss.cpu().item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%%'
                    #              % (100. * correct / total))
            print(f' - val_acc: {correct / total:.{5}}')
            # platue.step(correct)
            log['val_acc'].append(100. * correct / total)

            # Save checkpoint.
            acc = 100. * correct / total
            # print('Saving..')
            state = {
                'optimizer': optimizer.state_dict(),
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if acc > best_acc:
                torch.save(state, f'./checkpoint/try_{args.try_no}best.t7')
                best_acc = acc
                best_no = correct
            torch.save(state, f'./checkpoint/try_{args.try_no}temp.t7')
            with open('log/try_{}.json'.format(args.try_no), 'w') as wr:
                wr.write(log.__str__())


        for epoch in range(start_epoch, start_epoch + args.epochs[i]):
            scheduler.step()
            train(epoch)
            test(epoch)
            print(f'best: {best_acc}: {best_no}/{len(val_loader)*args.val_batch_size}')
        start_epoch += args.epochs[i]
