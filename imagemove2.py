import json
import os
import shutil

rootpath = '/media/palm/Unimportant/pdr2018'
dirs = ['AgriculturalDisease_validationset', 'AgriculturalDisease_trainingset']
filrs = ['AgriculturalDisease_validation_annotations.json',
         'AgriculturalDisease_train_annotations.json']
dests = ['validate', 'train']
for i in range(2):
    try:
        os.mkdir(f'{rootpath}/{dests[i]}')
    except:
        pass
    jsn = json.load(open(f'{rootpath}/{dirs[i]}/{filrs[i]}'))
    x = 0
    for item in jsn:
        if not os.path.isdir(f'{rootpath}/{dests[i]}/{item["disease_class"]:02}'):
            os.mkdir(f'{rootpath}/{dests[i]}/{item["disease_class"]:02}')
        shutil.copy(f'{rootpath}/{dirs[i]}/images/{item["image_id"]}'.encode('utf-8').strip(),
                    f'{rootpath}/{dests[i]}/{item["disease_class"]:02}/{item["image_id"]}'.encode('utf-8').strip())
