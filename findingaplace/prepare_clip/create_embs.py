import torch
import clip
from PIL import Image
import os
import pickle

# MODULE TO PRODUCE CLIP EMBEDDINGS OF YOUR DATASET
# FOR MORE INFO ABOUT CLIP https://github.com/openai/CLIP?source=post_page-----d3ba20c0068--------------------------------


def make_embs(p, embeddings_file):
    """
    produces a pickle file of CLIP embeddings of your dataset
    :param p: folder of (cropped) images
    :param embeddings_file: file to store embeddings
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device) # load CLIP model ViT-L/14
    try:
        images = [os.path.join(p, i) for i in os.listdir(p) if not i.endswith('json')]
    except NotADirectoryError:
        print(f'{p} should be a directory containing image files')
        return
    # Embedding of the input image
    embs = dict()
    with torch.no_grad():
        for image_name in images:  # for each image
            image = preprocess(Image.open(image_name)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)   # CLIP encode
            image_features /= torch.linalg.norm(image_features, dim=-1, keepdims=True)  # normalise
            embs[image_name] = image_features  # save in dictionary
    print(embs)
    f = open(embeddings_file, "wb")  # savo to pickle file
    pickle.dump(embs, f)
    f.close()

