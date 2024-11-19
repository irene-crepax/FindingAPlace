import os
import shutil
import json


def split_dir(folders):
    train_folder, val_folder = folders[0], folders[1]
    annotation_files = {'annotations-train.json': train_folder, 'annotations-val.json': val_folder}
    for annotation_file in annotation_files.keys():
        d = annotation_files[annotation_file]
        os.mkdir(d)
        file = open(annotation_file, 'r')
        annotations = json.load(file)
        images = annotations['images']
        for image in images:
            image_name = os.path.split(image)[1]
            shutil.copyfile(image_name, os.path.join(d, image_name))

