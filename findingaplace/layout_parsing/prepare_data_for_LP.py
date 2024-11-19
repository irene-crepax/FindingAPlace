import json
import os
import cv2


def create_empty_annotations(directory, annotations_file):
    """
    creates an empty annotations file in the COCO dataset style
    :param directory: folder containing images
    """
    images = list()
    i = 0
    try:
        #dir = os.listdir(directory)  # throw error if param is not a valid directory
        dir = list()
        for r, dirs, files in os.walk(directory):
            for file in files:
                dir.append(os.path.join(r, file))
    except NotADirectoryError:
        print(f'{directory} is not a directory')
        return
    for image in dir:  # iterate over images in folder
        try:
            #img = cv2.imread(os.path.join(directory, image))
            img = cv2.imread(image)
            h = img.shape[0]
            w = img.shape[1]
            images.append({"width": w, "height": h, "id": i, "file_name": image})    # create dictionary for each image
            i += 1
        except AttributeError:
            print(f'{image} is not an image file')
            pass

    json_object = json.dumps(images, indent=2)

    with open(os.path.join(annotations_file), "w") as outfile:   # save dictionary as annotations.json file
        outfile.write(json_object)
    outfile.close()
