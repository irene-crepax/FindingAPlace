import cv2
import pandas as pd
import json
import pytesseract
import os
import imutils

config = '--oem 3 --psm 6'


# CUSTOM OCR USING PYTESSERACT https://pypi.org/project/pytesseract/, A PYTHON WRAPPER FOR TESSERACT OCR https://github.com/tesseract-ocr/tesseract
def bruteforceocr(labels_path, images_path, output, ocr_labels, config=config):
    """
    ocr's captions identified using fine-tuned Layout Parser and saves text in a csv file with one image per row
    :param labels_path: path to json prediction files
    :param images_path: path to images
    :param output: csv file to store ocr
    :param config: configuration parameters for OCR engine
    """
    all_captions = dict()
    try:
        all_labels = os.listdir(labels_path)
    except NotADirectoryError:
        print(f'{labels_path} should be a valid directory containing label files ')
        return
    for label in all_labels:  # iterates over prediction files
        captions_list = list()
        captions_dict = dict()
        with open(os.path.join(labels_path, label)) as json_file:
            try:
                data = json.load(json_file)
                json_file.close()
            except AttributeError:
                print(f'{label} is not a valid json file; check that f{labels_path} is a folder containing labels as '
                      f'valid json files')
                return
                # img_path = label.replace('json', 'jpg')
            img_path = data['filename']
            try:
                #img = cv2.imread(os.path.join(images_path, img_path))
                img = cv2.imread(img_path)
            except AttributeError:
                print(f'{img_path} is not an image; check that f{images_path} is a folder containing image files')
                return
            i = 0
            for item in data['annotations']:  # iterates over annotations for each prediction file
                if item['label'] in ocr_labels:  # if the annotation is a caption
                    pts_list = [[int(pt) for pt in coords] for coords in item['points']]  # retrieve coordinates
                    x = [item[0] for item in pts_list]
                    y = [item[1] for item in pts_list]
                    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                    image = img[y1:y2, x1:x2]
                    confs = dict()
                    for angle in [0, 90, 180, 270]:  # for each possible rotation of the image
                        im = imutils.rotate_bound(image, angle)
                        text = pytesseract.image_to_data(im, config=config, lang='eng', output_type='data.frame')
                        text = text[text.conf != -1]
                        conf = text.groupby(['block_num'])['conf'].mean()  # calculate confidence of ocr
                        confs[angle] = conf.values
                        if confs[angle].size == 0:
                            confs[angle] = 0
                    confs_values = list(confs.values())
                    conf = max(confs_values)  # choose rotation angle with highest ocr confidence
                    angle = [i for i in confs if confs[i] == conf]
                    im = imutils.rotate_bound(image, angle=angle[0])
                    ocr_result = pytesseract.image_to_string(im, lang='eng')
                    captions_list.append(ocr_result)
                    captions_dict[len(captions_list)] = [ocr_result,
                                                         angle[0]]  # save ocr and rotation angle in a dictionary
                    i += 1
        all_captions[img_path] = captions_dict
    caption_dict = dict()
    for i, (page, features) in enumerate(
            all_captions.items()):  # convert dictionary of {image: {ocf dictionary}} to dataframe
        num_images = len(features)
        caption_dict[i] = dict()
        for j in range(num_images):
            caption_dict[i]["Page"] = page
            caption_dict[i]["text_{}".format(j + 1)] = features[j + 1][0]
            caption_dict[i]["rotation_{}".format(j + 1)] = features[j + 1][1]
    captions_df = pd.DataFrame.from_dict(caption_dict, orient="index")
    captions_df.to_csv(output, sep=",", index=False)  # save dataframe as a csv file
