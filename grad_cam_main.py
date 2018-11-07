import copy

import os
import cv2
import numpy as np
import torch
from imagemove2 import getlookup
from torchvision import models, transforms

from grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

# if model has LSTM
# torch.backends.cudnn.enabled = False


def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def getmodel(cls=61):
    # model_conv = models.nasnet.nasnetmobile(cls)
    model_conv = models.densenet.densenet201(pretrained=True)
    num_ftrs = model_conv.classifier.in_features
    # model_conv.classifier = nn.Conv1d(num_ftrs, cls, kernel_size=(1, 1))
    model_conv.classifier = torch.nn.Linear(num_ftrs, cls)
    return model_conv


def main():
    root_path = '/media/palm/Unimportant/pdr2018/typesep_validate/Tomato/'
    image_name = 'c9ebc74c2177ce60a8230855333fb9e7.jpg'
    folder_name = '14_Tomato_Spider_Mite_Damage_Serious'
    # image_path = root_path+'/14_Tomato_Spider_Mite_Damage_Serious/1c0f1ae1374d01c2933069232735a331.jpg'
    image_path = os.path.join(root_path, folder_name, image_name)
    topk = 1
    cuda = 'cuda'
    arch = 'densenet201'
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        # Add your model
    }.get(arch)
    a, b, c = getlookup()
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    # Synset words
    classes = c['Tomato']

    # Model
    model = getmodel(20)
    checkpoint = torch.load('checkpoint/try_4_densesep-Tomatotemp.t7')
    model.load_state_dict(checkpoint['net'])
    model.to('cuda')
    model.eval()

    # Image
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])(raw_image).unsqueeze(0)

    # =========================================================================
    print('Grad-CAM')
    # =========================================================================
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image.to(device))

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=CONFIG['target_layer'])

        save_gradcam('results/{}_{}_gcam_{}.png'.format(image_name, classes[idx[i]], arch), output, raw_image)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

    # =========================================================================
    print('Vanilla Backpropagation')
    # =========================================================================
    bp = BackPropagation(model=model)
    probs, idx = bp.forward(image.to(device))

    for i in range(0, topk):
        bp.backward(idx=idx[i])
        output = bp.generate()

        save_gradient('results/{}_{}_bp_{}.png'.format(image_name, classes[idx[i]], arch), output)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

    # =========================================================================
    print('Deconvolution')
    # =========================================================================
    deconv = Deconvolution(model=copy.deepcopy(model))  # TODO: remove hook func in advance
    probs, idx = deconv.forward(image.to(device))

    for i in range(0, topk):
        deconv.backward(idx=idx[i])
        output = deconv.generate()

        save_gradient('results/{}_{}_deconv_{}.png'.format(image_name, classes[idx[i]], arch), output)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

    # =========================================================================
    print('Guided Backpropagation/Guided Grad-CAM')
    # =========================================================================
    gbp = GuidedBackPropagation(model=model)
    probs, idx = gbp.forward(image.to(device))

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        region = gcam.generate(target_layer=CONFIG['target_layer'])

        gbp.backward(idx=idx[i])
        feature = gbp.generate()

        h, w, _ = feature.shape
        region = cv2.resize(region, (w, h))[..., np.newaxis]
        output = feature * region

        save_gradient('results/{}_{}_gbp_{}.png'.format(image_name, classes[idx[i]], arch), feature)
        save_gradient('results/{}_{}_ggcam_{}.png'.format(image_name, classes[idx[i]], arch), output)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))


if __name__ == '__main__':
    main()
