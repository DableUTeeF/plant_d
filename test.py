import torch.nn as nn
import torch.utils.data.distributed
from torchvision.models.densenet import densenet201
from torchvision import transforms
from torchvision import datasets
import json


def densenet(cls=61):
    model_conv = densenet201(pretrained=False)
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, cls)
    return model_conv


if __name__ == '__main__':
    model = densenet().cuda()
    # print(model)
    checkpoint = torch.load('/home/palm/PycharmProjects/plant_d/checkpoint/try_1_dense201best.t7')
    model.load_state_dict(checkpoint['net'])
    directory = '/home/palm/PycharmProjects/DATA/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/'
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
            normalize,
        ])),
        batch_size=1,
        num_workers=4,
        pin_memory=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            out.append({'image_id': val_loader.sampler.data_source.imgs[batch_idx][0].split('/')[-1],
                        'disease_class': int(predicted.cpu().detach().numpy()[0]),
                        })

            # correct += predicted.eq(targets).sum().item()
            c += 1
            print(correct, '/', c, end='\r')

    with open('/home/palm/PycharmProjects/plant_d/prd/1_dense201.json', 'w') as wr:
        json.dump(out, wr)
