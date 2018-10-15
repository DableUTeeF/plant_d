import torch.nn as nn
import torch.utils.data.distributed
from torchvision.models.densenet import densenet201
from torchvision import transforms
from torchvision import datasets
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from imagemove2 import getlookup


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')


def densenet(cls=61):
    model_conv = densenet201(pretrained=False)
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, cls)
    return model_conv


if __name__ == '__main__':
    lookup = getlookup()
    model = densenet(len(lookup[2]['Tomato'])).cuda()

    checkpoint = torch.load('/home/palm/PycharmProjects/plant_d/checkpoint/try_1_densesep-Tomatobest.t7')
    model.load_state_dict(checkpoint['net'])
    directory = '/media/palm/Unimportant/pdr2018/typesep_validate/Tomato'
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
            transforms.CenterCrop(224),
            transforms.ToTensor(),

            normalize,
        ])),
        batch_size=1,
        num_workers=4,
        pin_memory=False)
    cls = val_loader.sampler.data_source.class_to_idx
    print(cls)
    y_test, y_pred = [], []
    cfs = 0  # temp correct
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs.view(-1, 3, 224, 224))
            _, predicted = outputs.max(1)
            y_test.append(targets)
            y_pred.append(predicted.cpu().detach().numpy()[0])
            cfs += predicted.eq(targets).sum().item()

            if int(predicted.cpu().detach().numpy()[0]) != int(val_loader.sampler.data_source.imgs[batch_idx][1]):
                out.append({'image_id': val_loader.sampler.data_source.imgs[batch_idx][0].split('/')[-1],
                            'predicted': int(predicted.cpu().detach().numpy()[0]),
                            'expected': val_loader.sampler.data_source.imgs[batch_idx][1],
                            })
            else:
                correct += 1

            c += 1
            print(cfs, '/', c, end='\r')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=sorted(lookup[2]['Tomato']),
                          title='Plant types')

    plt.show()

