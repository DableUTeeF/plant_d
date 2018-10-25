r"""
    type: Apple: 00, Citrus: 01, Cedar: 02, Peach: 03, Pepper: 04,
          Tomato: 05, Strawberry: 06, Potato: 07, Corn: 08, Grape: 09, Cherry: 10
"""
import json
import os
import shutil


def convert(word):
    return ''.join(x.capitalize()+'_' or '_' for x in word.split('_'))[:-1]


def getlookup():
    a = {
        0: convert('Apple_Healthy'),
        1: convert('Apple_Scab_general'),
        2: convert('Apple_Scab_serious'),
        3: convert('Apple_Frogeye_Spot'),
        4: convert('Cedar_Apple_Rust_general'),
        5: convert('Cedar_Apple_Rust_serious'),
        6: convert('Cherry_Healthy'),
        7: convert('Cherry_Powdery_Mildew_general'),
        8: convert('Cherry_Powdery_Mildew_serious'),
        9: convert('Corn_Healthy'),
        10: convert('Corn_Cercospora_zeaemaydis_Tehon_and_Daniels_general'),
        11: convert('Corn_Cercospora_zeaemaydis_Tehon_and_Daniels_serious'),
        12: convert('Corn_Puccinia_polysora_general'),
        13: convert('Corn_Puccinia_polysora_serious'),
        14: convert('Corn_Curvularia_leaf_spot_fungus_general'),
        15: convert('Corn_Curvularia_leaf_spot_fungus_serious'),
        16: convert('Corn_Maize_dwarf_mosaic_virus'),
        17: convert('Grape_Heathy'),
        18: convert('Grape_Black_Rot_Fungus_general'),
        19: convert('Grape_Black_Rot_Fungus_serious'),
        20: convert('Grape_Black_Measles_Fungus_general'),
        21: convert('Grape_Black_Measles_Fungus_serious'),
        22: convert('Grape_Leaf_Blight_Fungus_general'),
        23: convert('Grape_Leaf_Blight_Fungus_serious'),
        24: convert('Citrus_Healthy'),
        25: convert('Citrus_Greening_June_general'),
        26: convert('Citrus_Greening_June_serious'),
        27: convert('Peach_Healthy'),
        28: convert('Peach_Bacterial_Spot_general'),
        29: convert('Peach_Bacterial_Spot_serious'),
        30: convert('Pepper_Healthy'),
        31: convert('Pepper_Scab_general'),
        32: convert('Pepper_Scab_serious'),
        33: convert('Potato_Healthy'),
        34: convert('Potato_Early_Blight_Fungus_general'),
        35: convert('Potato_Early_Blight_Fungus_serious'),
        36: convert('Potato_Late_Blight_Fungus_general'),
        37: convert('Potato_Late_Blight_Fungus_serious'),
        38: convert('Strawberry_Healthy'),
        39: convert('Strawberry_Scorch general'),
        40: convert('Strawberry_Scorch serious'),
        41: convert('Tomato_Healthy'),
        42: convert('Tomato_Powdery_mildew_general'),
        43: convert('Tomato_Powdery_mildew_serious'),
        44: convert('Tomato_Bacterial_Spot_Bacteria_general'),
        45: convert('Tomato_Bacterial_Spot_Bacteria_serious'),
        46: convert('Tomato_Early_Blight_Fungus_general'),
        47: convert('Tomato_Early_Blight_Fungus_serious'),
        48: convert('Tomato_Late_Blight_Water_Mold_general'),
        49: convert('Tomato_Late_Blight_Water_Mold_serious'),
        50: convert('Tomato_Leaf_Mold_Fungus_general'),
        51: convert('Tomato_Leaf_Mold_Fungus_serious'),
        52: convert('Tomato_Target_Spot_Bacteria_general'),
        53: convert('Tomato_Target_Spot_Bacteria_serious'),
        54: convert('Tomato_Septoria_Leaf_Spot_Fungus_general'),
        55: convert('Tomato_Septoria_Leaf_Spot_Fungus_serious'),
        56: convert('Tomato_Spider_Mite_Damage_general'),
        57: convert('Tomato_Spider_Mite_Damage_serious'),
        58: convert('Tomato_YLCV_Virus_general'),
        59: convert('Tomato_YLCV_Virus_serious'),
        60: convert('Tomato_Tomv')}
    # a = {
    #     0: 'Apple_healthy',
    #     1: 'Apple_Scab_general',
    #     2: 'Apple_Scab_serious',
    #     3: 'Apple_Frogeye_Spot',
    #     4: 'Cedar_Apple_Rust_general',
    #     5: 'Cedar_Apple_Rust_serious',
    #     6: 'Cherry_healthy',
    #     7: 'Cherry_Powdery_Mildew_general',
    #     8: 'Cherry_Powdery_Mildew_serious',
    #     9: 'Corn_healthy',
    #     10: 'Corn_Cercospora_zeaemaydis_Tehon_and_Daniels_general',
    #     11: 'Corn_Cercospora_zeaemaydis_Tehon_and_Daniels_serious',
    #     12: 'Corn_Puccinia_polysora_general',
    #     13: 'Corn_Puccinia_polysora_serious',
    #     14: 'Corn_Curvularia_leaf_spot_fungus_general',
    #     15: 'Corn_Curvularia_leaf_spot_fungus_serious',
    #     16: 'Corn_Maize_dwarf_mosaic_virus',
    #     17: 'Grape_heathy',
    #     18: 'Grape_Black_Rot_Fungus_general',
    #     19: 'Grape_Black_Rot_Fungus_serious',
    #     20: 'Grape_Black_Measles_Fungus_general',
    #     21: 'Grape_Black_Measles_Fungus_serious',
    #     22: 'Grape_Leaf_Blight_Fungus_general',
    #     23: 'Grape_Leaf_Blight_Fungus_serious',
    #     24: 'Citrus_healthy',
    #     25: 'Citrus_Greening_June_general',
    #     26: 'Citrus_Greening_June_serious',
    #     27: 'Peach_healthy',
    #     28: 'Peach_Bacterial_Spot_general',
    #     29: 'Peach_Bacterial_Spot_serious',
    #     30: 'Pepper_healthy',
    #     31: 'Pepper_scab_general',
    #     32: 'Pepper_scab_serious',
    #     33: 'Potato_healthy',
    #     34: 'Potato_Early_Blight_Fungus_general',
    #     35: 'Potato_Early_Blight_Fungus_serious',
    #     36: 'Potato_Late_Blight_Fungus_general',
    #     37: 'Potato_Late_Blight_Fungus_serious',
    #     38: 'Strawberry_healthy',
    #     39: 'Strawberry_Scorch general',
    #     40: 'Strawberry_Scorch serious',
    #     41: 'Tomato_healthy',
    #     42: 'Tomato_powdery_mildew_general',
    #     43: 'Tomato_powdery_mildew_serious',
    #     44: 'Tomato_Bacterial_Spot_Bacteria_general',
    #     45: 'Tomato_Bacterial_Spot_Bacteria_serious',
    #     46: 'Tomato_Early_Blight_Fungus_general',
    #     47: 'Tomato_Early_Blight_Fungus_serious',
    #     48: 'Tomato_Late_Blight_Water_Mold_general',
    #     49: 'Tomato_Late_Blight_Water_Mold_serious',
    #     50: 'Tomato_Leaf_Mold_Fungus_general',
    #     51: 'Tomato_Leaf_Mold_Fungus_serious',
    #     52: 'Tomato_Target_Spot_Bacteria_general',
    #     53: 'Tomato_Target_Spot_Bacteria_serious',
    #     54: 'Tomato_Septoria_Leaf_Spot_Fungus_general',
    #     55: 'Tomato_Septoria_Leaf_Spot_Fungus_serious',
    #     56: 'Tomato_Spider_Mite_Damage_general',
    #     57: 'Tomato_Spider_Mite_Damage_serious',
    #     58: 'Tomato_YLCV_Virus_general',
    #     59: 'Tomato_YLCV_Virus_serious',
    #     60: 'Tomato_Tomv'}
    b = list(set([a[f].split('_')[0] for f in a]))
    c = {}
    for idx in a:
        ci = b.index(a[idx].split('_')[0])
        if b[ci] not in c:
            c[b[ci]] = []
        c[b[ci]].append(a[idx])
    for k in c:
        c[k] = sorted(c[k])
    return a, sorted(b), c


def getlookup2():
    a, b, c = getlookup()
    for key in c:
        plant = c[key]
        for idx, disese in enumerate(plant):
            x = disese.find('_serious') + 1 + disese.find('_general')
            if x != -1:
                plant[idx] = disese[:x]
    for idx in a:
        x = a[idx].find('_serious') + 1 + a[idx].find('_general')
        if x != -1:
            a[idx] = a[idx][:x]
    return a, c


def move():
    rootpath = '/root/palm/DATA/plant'
    # rootpath = '/media/palm/Unimportant/pdr2018'
    dirs = ['AgriculturalDisease_validationset', 'AgriculturalDisease_trainingset']
    filrs = ['AgriculturalDisease_validation_annotations.json',
             'AgriculturalDisease_train_annotations.json']
    dests = ['typesep_validate', 'typesep_train']
    a, b, c = getlookup()
    d = [a[x] for x in a]
    for i in range(2):
        try:
            os.mkdir(os.path.join(rootpath, dests[i]))
        except:
            pass
        for type_p in b:
            try:
                os.mkdir(os.path.join(rootpath, dests[i], type_p))
            except:
                pass
            for disease in c[type_p]:
                didx = c[type_p].index(disease)
                disease = f'{didx:02}_{disease}'
                try:
                    os.mkdir(os.path.join(rootpath, dests[i], type_p, disease))
                except:
                    pass

    for i in range(2):
        jsn = json.load(open(f'{rootpath}/{dirs[i]}/{filrs[i]}'))
        for item in jsn:
            type_p = a[item["disease_class"]].split('_')[0]
            filename = item["image_id"]
            disease = a[item["disease_class"]]
            didx = c[type_p].index(disease)
            disease = f'{didx:02}_{disease}'
            shutil.copy(f'{rootpath}/{dirs[i]}/images/{item["image_id"]}'.encode('utf-8').strip(),
                        os.path.join(rootpath, dests[i], type_p, disease, filename).encode('utf-8').strip())


def move2():
    # rootpath = '/root/palm/DATA/plant'
    rootpath = '/media/palm/Unimportant/pdr2018'
    dirs = ['AgriculturalDisease_validationset', 'AgriculturalDisease_trainingset']
    filrs = ['AgriculturalDisease_validation_annotations.json',
             'AgriculturalDisease_train_annotations.json']
    dests = ['typesep_type_validate', 'typesep_type_train']
    a, b, c = getlookup()
    for i in range(2):
        try:
            os.mkdir(os.path.join(rootpath, dests[i]))
        except:
            pass
        for type_p in b:
            tidx = b.index(type_p)
            try:
                os.mkdir(os.path.join(rootpath, dests[i], type_p))
            except:
                pass

    for i in range(2):
        jsn = json.load(open(f'{rootpath}/{dirs[i]}/{filrs[i]}'))
        count = {}
        for item in jsn:
            type_p = a[item["disease_class"]].split('_')[0]
            filename = item["image_id"]
            tidx = b.index(type_p)
            if type_p not in count:
                count[type_p] = 0
            count[type_p] += 1
            _ = shutil.copy(f'{rootpath}/{dirs[i]}/images/{item["image_id"]}'.encode('utf-8').strip(),
                        os.path.join(rootpath, dests[i], type_p, filename).encode('utf-8').strip())
        print(count)


def move3():
    rootpath = '/media/palm/Unimportant/pdr2018'
    dirs = ['AgriculturalDisease_validationset', 'AgriculturalDisease_trainingset']
    filrs = ['AgriculturalDisease_validation_annotations.json',
             'AgriculturalDisease_train_annotations.json']
    dests = ['typesep_dup_validate', 'typesep_dup_train']
    a, b, c = getlookup()
    d, e = getlookup2()
    for i in range(2):
        try:
            os.mkdir(os.path.join(rootpath, dests[i]))
        except:
            pass
        for type_p in b:
            try:
                os.mkdir(os.path.join(rootpath, dests[i], type_p))
            except:
                pass
            for disease in d[type_p]:
                didx = d[type_p].index(disease)
                try:
                    os.mkdir(os.path.join(rootpath, dests[i], type_p, disease))
                except:
                    pass

    for i in range(2):
        jsn = json.load(open(f'{rootpath}/{dirs[i]}/{filrs[i]}'))
        for item in jsn:
            type_p = d[item["disease_class"]].split('_')[0]
            filename = item["image_id"]
            disease = d[item["disease_class"]]
            didx = d[type_p].index(disease)
            shutil.copy(f'{rootpath}/{dirs[i]}/images/{item["image_id"]}'.encode('utf-8').strip(),
                        os.path.join(rootpath, dests[i], type_p, disease, filename).encode('utf-8').strip())


def x():
    rootpath = '/media/palm/Unimportant/pdr2018'

    dirs = ['AgriculturalDisease_validationset', 'AgriculturalDisease_trainingset']
    filrs = ['AgriculturalDisease_validation_annotations.json',
             'AgriculturalDisease_train_annotations.json']
    a, b, c = getlookup()
    d = {}
    e = 0
    for i in range(2):
        jsn = json.load(open(f'{rootpath}/{dirs[i]}/{filrs[i]}'))
        for item in jsn:
            type_p = a[item["disease_class"]].split('_')[0]
            filename = item["image_id"]
            disease = a[item["disease_class"]]
            didx = c[type_p].index(disease)
            if type_p not in d:
                d[type_p] = {}
            if disease not in d[type_p]:
                d[type_p][disease] = 0
            if 'total' not in d[type_p]:
                d[type_p]['total'] = 0
            d[type_p]['total'] += 1
            d[type_p][disease] += 1
            e += 1
    return d, e


if __name__ == '__main__':
    # move2()
    # move()
    pass
