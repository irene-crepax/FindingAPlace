import json
import os
import cv2


def load_categories_as_dict(categories_as_string):
    return {k: v for v, k in enumerate(categories_as_string)}


def retrieve_annotations(jfilepath, index, j, file_path, catdict):
    jfile = open(file_path + jfilepath)
    data = json.load(jfile)
    img_annotations = list()
    for shape in data['shapes']:
        pts_list = [[int(pt) for pt in coords] for coords in shape['points']]
        x = [item[0] for item in pts_list]
        y = [item[1] for item in pts_list]
        x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
        w = x2 - x1
        h = y2 - y1
        img_annotations.append(
            {"id": j, "image_id": index, "category_id": catdict[shape['label'].replace(" ", "")], "segmentation": [],
             "bbox": [x1, y1, x2, y2], "ignore": 0, "iscrowd": 0, "area": w * h})
        j += 1
    return img_annotations, j


# categories = [{"id": 0, "name": "caption"}, {"id": 1, "name": "text"}, {"id": 2, "name": "figure"}, {"id": 3, "name":"title"}]

def convert(filepath, cats):
    cat_dict = load_categories_as_dict(cats)
    categories = list()
    for cat in cat_dict:
        d = {"id": cat, "name": cat_dict[cat]}
        categories.append(d)
    images = list()
    annotations = list()
    i = 0
    j = 0
    for file in os.listdir(filepath):
        if file.endswith('.jpg'):
            img = cv2.imread(filepath + file)
            h = img.shape[0]
            w = img.shape[1]
            images.append({"width": w, "height": h, "id": i, "file_name": file})
        else:
            anns, j = retrieve_annotations(file, i, j)
            annotations.extend(anns)
            i += 1

    coco_dictionary = {"images": images, "categories": categories, "annotations": annotations}
    print(coco_dictionary)
    json_object = json.dumps(coco_dictionary, indent=2)

    with open("finetuned_results.json", "w") as outfile:
        outfile.write(json_object)
