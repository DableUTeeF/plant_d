import json
import os
import shutil

# rootpath = '/media/palm/Unimportant/pdr2018'
rootpath = '/root/palm/DATA/plant'
dirs = ['AgriculturalDisease_validationset', 'AgriculturalDisease_trainingset']
filrs = ['ai_challenger_pdr2018_validation_annotations_20181021.json',
         'ai_challenger_pdr2018_train_annotations_20181021.json']
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
