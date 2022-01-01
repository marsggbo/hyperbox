import json
import os
from glob import glob

from hyperbox_app.covid19.datasets.utils import pil_loader


def statistics_slice_size(jsonfile, path):
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    statistics_num_patient_scan(data)
    info = {
        'min_slices': 9999,
        'max_slices': -1,
        'min_height': 9999,
        'max_height': -1,
        'min_width': 9999,
        'max_width': -1,
    }
    count = 0
    for _cls in data:
        count += 1
        if count>10:
            break
        for _patient_id in data[_cls]:
            for _scan_id in data[_cls][_patient_id]:
                slice_num = len(data[_cls][_patient_id][_scan_id])
                if slice_num<10:
                    print(f"{_cls}_{_patient_id}_{_scan_id}: {slice_num}")
                if slice_num > info['max_slices']:
                    info['max_slices'] = slice_num
                    info['max_slices_name'] = f"{_cls}_{_patient_id}_{_scan_id}"
                if slice_num < info['min_slices']:
                    info['min_slices'] = slice_num
                    info['min_slices_name'] = f"{_cls}_{_patient_id}_{_scan_id}"
                for i, filename in enumerate(data[_cls][_patient_id][_scan_id][:5]):
                    file = os.path.join(path, _cls, _patient_id, _scan_id, filename)
                    img = pil_loader(file)
                    height, width = img.size
                    if height > info['max_height']:
                        info['max_height'] = height
                        info['max_height_name'] = f"{_cls}_{_patient_id}_{_scan_id}"
                    if height < info['min_height']:
                        info['min_height'] = height
                        info['min_height_name'] = f"{_cls}_{_patient_id}_{_scan_id}"
                    if width > info['max_width']:
                        info['max_width'] = width
                        info['max_width_name'] = f"{_cls}_{_patient_id}_{_scan_id}"
                    if width < info['min_width']:
                        info['min_width'] = width
                        info['min_width_name'] = f"{_cls}_{_patient_id}_{_scan_id}"
    print(info)
    return info

def statistics_num_patient_scan(data):
    info = {}
    for c in data:
        info[f'{c}_#patient'] = len(data[c])
        info[f'{c}_#scan'] = 0
        for p in data[c]:
            info[f'{c}_#scan'] += len(data[c][p])
    print(info)


if __name__ == '__main__':
    # ccccii
    statistics_slice_size(
        '/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/ccccii/ct_test.json',
        '/home/datasets/CCCCI_cleaned/dataset_cleaned/'
    )
    # # iran
    # statistics_slice_size(
    #     '/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/iran/train.json',
    #     '/home/datasets/COVID-CTset_visual'
    # )
    # mosmed
    # statistics_slice_size(
    #     '/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/mosmed/nii_png_train.json',
    #     '/home/datasets/MosMedData/COVID19_1110/pngs'
    # )