import torch.nn as nn
import torch.utils.data.distributed
from torchvision.models.densenet import densenet201
from torchvision import transforms
from torchvision import datasets
import json
from imagemove2 import getlookup

a, b, c = getlookup()
d = [a[x] for x in a]
labels_1_ptype = {0: 8,
                  1: 2,
                  2: 0,
                  3: 7,
                  4: 9,
                  5: 1,
                  6: 4,
                  7: 3,
                  8: 10,
                  9: 5,
                  10: 6,
                  }
startidx = {'Apple': 0,
            'Cedar': 4,
            'Cherry': 6,
            'Corn': 9,
            'Grape': 17,
            'Citrus': 24,
            'Peach': 27,
            'Pepper': 30,
            'Potato': 33,
            'Strawberry': 38,
            'Tomato': 41
            }


def densenet(cls=61):
    model_conv = densenet201(pretrained=False)
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, cls)
    return model_conv


def predict(x, cls):
    ncls = len(c[b[cls]])
    plant = b[cls]

    model = densenet(ncls).cuda()
    checkpoint = torch.load(f'/home/palm/PycharmProjects/plant_d/checkpoint/try_3_densesep-{plant}temp.t7')
    model.load_state_dict(checkpoint['net'])

    return model(x), plant


def getidx(plant, idx):
    return d.index(c[plant][idx])


if __name__ == '__main__':
    mf = []
    for idx, plant in enumerate(b):
        ncls = len(c[plant])
        # mm = densenet(ncls).cuda()
        mf.append(densenet(ncls).cuda())
        checkpoint = torch.load(f'checkpoint/try_3_densesep-{plant}best.t7')
        mf[idx].load_state_dict(checkpoint['net'])
    model = densenet(11).cuda()
    # print(model)
    checkpoint = torch.load('/home/palm/PycharmProjects/plant_d/checkpoint/try_2_denseptype-temp.t7')
    model.load_state_dict(checkpoint['net'])
    directory = '/home/palm/PycharmProjects/DATA/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/'
    out = []
    co = 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    model.eval()
    correct = 0
    total = 0
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(directory, transforms.Compose([
            transforms.Resize(256),
            # ReplicatePad(args.imsize_l[i]),
            transforms.CenterCrop(224),
            # transforms.FiveCrop(224),

            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1,
        num_workers=4,
        pin_memory=False)

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(val_loader):
            inputs, _ = inputs.to('cuda'), _.to('cuda')
            x = model(inputs)
            _, predicted = x.max(1)
            cls = predicted.cpu().detach().numpy()[0]
            ncls = len(c[b[cls]])
            plant = b[cls]

            fff = mf[cls](inputs)
            _, predicted = fff.max(1)
            # label = startidx[plant] + predicted.cpu().detach().numpy()[0]
            label = getidx(plant, predicted.cpu().detach().numpy()[0])
            out.append({'image_id': val_loader.sampler.data_source.imgs[batch_idx][0].split('/')[-1],
                        'disease_class': label,
                        })

            # correct += predicted.eq(targets).sum().item()
            co += 1
            print(co, end='\r')

    with open('/home/palm/PycharmProjects/plant_d/prd/2_sep.json', 'w') as wr:
        json.dump(out, wr)
