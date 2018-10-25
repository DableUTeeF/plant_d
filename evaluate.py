import torch.nn as nn
import torch.utils.data.distributed
from torchvision.models.densenet import densenet201
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import numpy as np
import os
from torchvision import datasets
from PIL import Image
import json


def densenet(cls=61):
    model_conv = densenet201(pretrained=False)
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, cls)
    return model_conv


if __name__ == '__main__':
    model = densenet(11).cuda()
    # print(model)
    checkpoint = torch.load('checkpoint/try_2_denseptype-temp.t7')
    model.load_state_dict(checkpoint['net'])
    # directory = '/root/palm/DATA/plant/typesep_type_validate/'
    directory = '/media/palm/Unimportant/pdr2018/typesep_type_validate'
    out = []
    c = 0
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
            # transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),

            normalize,
        ])),
        batch_size=1,
        num_workers=4,
        pin_memory=False)
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
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            null = int(predicted.cpu().detach().numpy()[0]), int(val_loader.sampler.data_source.imgs[batch_idx][1])
            if int(predicted.cpu().detach().numpy()[0]) != int(val_loader.sampler.data_source.imgs[batch_idx][1]):
                out.append({'image_id': val_loader.sampler.data_source.imgs[batch_idx][0].split('/')[-1],
                            'predicted': int(predicted.cpu().detach().numpy()[0]),
                            'expected': val_loader.sampler.data_source.imgs[batch_idx][1],
                            })
            else:
                correct += 1
            c += 1
            print(correct, '/', c, '/', f'{null}', end='\r')

    with open('False_prd_1_dense201.json', 'w') as wr:
        json.dump(out, wr)
