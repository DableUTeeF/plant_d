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


def predict(inputs, cls):
    ncls = len(c[b[cls]])
    plant = b[cls]

    mm = densenet(ncls).cuda()
    checkpoint = torch.load(f'checkpoint/try_3_densesep-{plant}best.t7')
    mm.load_state_dict(checkpoint['net'])
    fff = mm(inputs)
    _, fsa = fff.max(1)
    return fsa, plant


def getidx(plant, idx):
    return d.index(c[plant][idx])


if __name__ == '__main__':
    out = []
    for key in c:
        model = densenet(len(c[key])).cuda()

        checkpoint = torch.load(f'/home/palm/PycharmProjects/plant_d/checkpoint/try_3_densesep-{key}temp.t7')
        model.load_state_dict(checkpoint['net'])
        # directory = '/root/palm/DATA/plant/validate/'
        directory = '/media/palm/Unimportant/pdr2018/validate/'
        co = 0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        model.eval()
        correct = 0
        total = 0
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(directory, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),

                normalize,
            ])),
            shuffle=True,
            batch_size=1,
            num_workers=4,
            pin_memory=False)
        samples = json.load(open('tmp/1_sepeval.json', 'r'))
        # model = mf[10]
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if val_loader.sampler.data_source.imgs[batch_idx][0] in samples[key]:
                    inputs, _ = inputs.to('cuda'), targets.to('cuda')
                    # x = model(inputs)
                    # _, pp = x.max(1)
                    cls = 10
                    ncls = len(c[b[cls]])
                    plant = b[cls]

                    x = model(inputs)
                    _, predicted = x.max(1)

                    # predicted, plant = predict(inputs, labels_1_ptype[pp.cpu().detach().numpy()[0]])

                    # label = startidx[plant] + predicted.cpu().detach().numpy()[0]
                    label = getidx(plant, predicted.cpu().detach().numpy()[0])
                    out.append({'image_id': val_loader.sampler.data_source.imgs[batch_idx][0].split('/')[-1],
                                'predicted': int(label),
                                'expected': int(targets.cpu().detach().numpy()[0]),
                                'plant': plant,
                                'output_from_model': int(predicted.cpu().detach().numpy()[0]),
                                })

                    correct += label == targets.cpu().detach().numpy()[0]
                    co += 1
                    print(f'{(correct/co):.2}', ': ', correct, '/', co, '            ', end='\r')

                    with open('tmp/1_sepeval_2.json', 'w') as wr:
                        json.dump(out, wr)
                    print(key, correct/co)
