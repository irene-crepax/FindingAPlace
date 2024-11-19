import json
import os
import cv2


def retrieve_cropped_images(labels_path, images_path, cropped_path, category):
    """
    crop illustrations from original images to create your CLIP search database
    :param labels_path: folder of annotations files
    :param images_path: folder of images
    :param cropped_path: folder to store cropped illustrations
    """
    try:
        labels = [os.path.join(labels_path, i) for i in os.listdir(labels_path)]
    except NotADirectoryError:
        print(f'{labels_path} is not a directory')
        return
    for label in labels:   # iterate over annotations file
        try:
            captions_label = json.load(open(label, 'r'))
        except AttributeError:
            print(f'{label} is not a valid json file; check the directory f{labels_path}')
            return
        try:
            #image_name = os.path.join(images_path, captions_label['filename'])
            image_name = captions_label['filename']
        except NotADirectoryError:
            print(f'{images_path} is not a valid directory')
            return
        annos = captions_label['annotations']
        i = 1
        for ann in annos:   # for each bounding box in the annotations
            if ann["label"] == category:    # if it is a figure
                image = cv2.imread(image_name)
                pts_list = [[int(pt) for pt in coords] for coords in ann['points']]  # retrieve coordinates
                x = [item[0] for item in pts_list]
                y = [item[1] for item in pts_list]
                x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                try:
                    cropped_image = image[y1:y2, x1:x2]  # crop image
                except TypeError:
                    print(f'{image} is not a valid image file; check f{images_path}')
                    return
                try:
                    name = captions_label['filename'].rsplit('\\', 1)[1]
                    cropped_name = os.path.join(cropped_path, '_'.join([str(i), name]))
                except NotADirectoryError:
                    print(f'{cropped_path} is not a valid directory')
                    return
                i += 1
                cv2.imwrite(cropped_name, cropped_image) # save cropped image


