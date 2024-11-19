import os

import torch
import clip
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import ast


# MODULE TO PERFORM CLIP SEARCH THROUGH YOUR EMBEDDINGS

def create_lists(l):
    new_l = list()
    for x in l:
        if len(x) != 0:
            new_l.append(ast.literal_eval(x))
        else:
            new_l.append([])
    return new_l


def clip_search(target, emb, img2img_search, output_file, num_results=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device) # load CLIP model ViT-L/14
    cos = torch.nn.CosineSimilarity(dim=0)
    with torch.no_grad():
        result = {}
        if img2img_search == 'image':
            image = preprocess(Image.open(target)).unsqueeze(0).to(device)
            features = model.encode_image(image)
            features /= torch.linalg.norm(features, dim=-1, keepdims=True)
        else:
            text = clip.tokenize(target).to(device)
            features = model.encode_text(text)

        for i, img in emb.items():
            sim = cos(img[0], features[0]).item()
            sim = (sim + 1) / 2
            result[i] = sim

        sorted_value = sorted(result.items(), key=lambda x: x[1],
                              reverse=True)  # sort embeddings by similarity score
        sorted_res = dict(sorted_value)
        top = list(sorted_res.keys())[:num_results]
        images_found = top
        images_found = [os.path.split(image)[1] for image in images_found]
        images_found = ['_'.join(image.split('_')[1:]) for image in images_found]
        df = pd.DataFrame(data={"Page": images_found})
        df.to_csv(output_file, sep=',',index=False)


