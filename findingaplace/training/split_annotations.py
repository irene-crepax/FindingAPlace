import json
import random


def load_categories_as_dict(categories_as_string):
    return {k: v for v, k in enumerate(categories_as_string)}


def split(cats):
    cat_dict = load_categories_as_dict(cats)
    file = open('finetuned_results.json', 'r')

    jobject = json.load(file)

    images = jobject['images']
    annotations = jobject['annotations']
    image_dict = dict()
    for idx, item in enumerate(images):
        image_dict[idx] = item
    test = 90

    # Random Sample Training and Test Data
    # Using keys() + randint() + computations
    key_list = list(image_dict.keys())

    test_key_count = int((len(key_list) / 100) * test)
    test_keys = sorted([random.choice(key_list) for ele in range(test_key_count)])
    train_keys = sorted([ele for ele in key_list if ele not in test_keys])
    testing_dict_images = dict((key, image_dict[key]) for key in test_keys
                               if key in image_dict)
    training_dict_images = dict((key, image_dict[key]) for key in train_keys
                                if key in image_dict)

    categories = list()
    for cat in cat_dict:
        d = {"id": cat, "name": cat_dict[cat]}
        categories.append(d)
    # categories = [{"id": 0, "name": "caption"}, {"id": 1, "name": "text"}, {"id": 2, "name": "figure"}, {"id": 3, "name":"title"}]

    imgs_training = list(training_dict_images.values())
    annos_training = list()
    for item in imgs_training:
        image_id = item['id']
        for anno in annotations:
            if anno['image_id'] == image_id:
                annos_training.append(anno)
    coco_dictionary_training = {"images": imgs_training, "categories": categories,
                                "annotations": annos_training}
    # print(coco_dictionary_training)
    json_object = json.dumps(coco_dictionary_training, indent=2)

    with open("annotations-train.json", "w") as outfile:
        outfile.write(json_object)
    imgs_testing = list(testing_dict_images.values())
    annos_testing = list()
    for item in imgs_testing:
        image_id = item['id']
        for anno in annotations:
            if anno['image_id'] == image_id:
                annos_testing.append(anno)
    coco_dictionary_testing = {"images": imgs_testing, "categories": categories,
                               "annotations": annos_testing}

    json_object = json.dumps(coco_dictionary_testing, indent=2)

    with open("annotations-val.json", "w") as outfile:
        outfile.write(json_object)
