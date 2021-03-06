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
from utils import Logger, format_time
import models
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import autoaugment as aug
import time


# todo: 0). Try https://github.com/DeepVoltaire/AutoAugment
# todo: 1). DenseNet201 with Conv1D as output. Dense6
# todo: 2). NasNetMobile with droppath. Nas1
# todo: 3). If 2nd goes well, implement droppath to another models.


def getmodel(cls=61):
    # model_conv = models.dense_nonorm.densenet201(pretrained=False)
    # num_ftrs = model_conv.classifier.in_features
    # model_conv.classifier = nn.Linear(num_ftrs, cls)
    model_conv = models.pnasnet.pnasnet5large(cls, 'imagenet')
    num_ftrs = model_conv.last_linear.in_features
    model_conv.last_linear = nn.Linear(num_ftrs, cls)
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

    args = DotDict({
        'batch_size': 6,
        'batch_mul': 4,
        'val_batch_size': 10,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': [180],
        'try_no': '3_pnasnet',
        'imsize': [331],
        'imsize_l': [350],
        'traindir': '/root/palm/DATA/plant/train',
        'valdir': '/root/palm/DATA/plant/validate',
        'workers': 16,
        'resume': False,
    })
    logger = Logger(f'./logs/{args.try_no}')
    logger.text_summary('Describe', 'Progressive NasNet5', 0)
    logger.text_summary('Describe', 'Batch size: 6*4', 1)
    logger.text_summary('Describe', 'Input size: 331/350', 2)
    logger.text_summary('Describe', 'Input size: 331/350', 3)
    best_acc = 0
    best_no = 0
    start_epoch = 1
    try:
        print(f'loading log: log/try_{args.try_no}.json')
        log = eval(open(f'log/try_{args.try_no}.json', 'r').read())
    except FileNotFoundError:
        log = {'acc': [], 'loss': [], 'val_acc': []}
        print(f'Log {args.try_no} not found')
    model = getmodel(61).cuda()
    # model = models.timedistdensenet.TimeDistDense201(True, 61).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # loss, acc, val_acc:
    # 128   0.001: [4.07, 7.33, ~8], 0.01: []
    # 32    0.001: [], 0.01: []
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                # nesterov=False,
                                )
    # scheduler = ExponentialLR(optimizer, 0.97)
    scheduler = MultiStepLR(optimizer, [30, 60, 90, 120, 150])
    # scheduler2 = MultiStepLR(optimizer, [2], gamma=5)
    # platue = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.2)
    criterion = nn.CrossEntropyLoss().cuda()
    zz = 0
    for i in range(len(args.epochs)):
        train_dataset = datasets.ImageFolder(
            args.traindir,
            transforms.Compose([
                transforms.Resize(args.imsize[i]),
                # ReplicatePad(args.imsize_l[i]),
                transforms.RandomResizedCrop(args.imsize[i]),
                aug.HandCraftPolicy(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.FiveCrop(args.imsize[i]),
                # transforms.Lambda(
                #     lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
                transforms.ToTensor(),
                # aug.Cutout(n_holes=1, length=20),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.valdir, transforms.Compose([
                transforms.Resize(args.imsize[i]),
                # ReplicatePad(args.imsize_l[i]),
                transforms.CenterCrop(args.imsize[i]),
                transforms.ToTensor(),
                # transforms.FiveCrop(args.imsize[i]),
                # transforms.Lambda(
                #     lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),

                normalize,
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
                # targets = torch.cat((targets, targets, targets, targets, targets))

                # bs, ncrops, c, h, w = inputs.size()
                # outputs = outputs.view(bs, ncrops, -1).mean(1)

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
                    print(f'\r{" "*(len(lss))}', end='')
                except NameError:
                    pass
                lss = f'{batch_idx}/{len(trainloader)} | ' + \
                      f'ETA: {format_time(step_time*(len(trainloader)-batch_idx))} - ' + \
                      f'loss: {train_loss/(batch_idx+1):.{3}} - ' + \
                      f'acc: {correct/total:.{5}}'
                print(f'\r{lss}', end='')

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
            logger.scalar_summary('acc', correct/total, epoch)
            logger.scalar_summary('loss', train_loss/(batch_idx+1), epoch)
            print(f'\r '
                  f'ToT: {format_time(time.time() - start_time)} - '
                  f'loss: {train_loss/(batch_idx+1):.{3}} - '
                  f'acc: {correct/total:.{5}}', end='')
            optimizer.step()
            optimizer.zero_grad()
            # scheduler2.step()
            log['acc'].append(100. * correct / total)
            log['loss'].append(train_loss / (batch_idx + 1))


        def test(epoch):
            global best_acc, best_no
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    """"""
                    # bs, ncrops, c, h, w = inputs.size()
                    # outputs = outputs.view(bs, ncrops, -1).mean(1)
                    """"""
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%%'
                    #              % (100. * correct / total))
            logger.scalar_summary('val_acc', correct/total, epoch)
            logger.scalar_summary('val_loss', test_loss/(batch_idx+1), epoch)
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
