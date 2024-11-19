# PIPELINE FOR CLIP SEARCH #

from prepare_clip import crop_images
from prepare_clip import create_embs
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str,
                    help='path to original images')
parser.add_argument('--labels_path', type=str,
                    help='path to folder to store layout files')
parser.add_argument('--cropped', type=str,
                    help='path to store image files of illustrations cropped out from original images')
parser.add_argument('--embeddings_file', type=str,
                    help='file to CLIP embeddings')
parser.add_argument('--category', type=str, default='figure',
                    help='type of visual element you wish to isolate')
args = parser.parse_args()
images_path = args.images_path  # path to folder of images to parse
labels_path = args.labels_path  # path to folder to store layout files
cropped = args.cropped  # path to folder to store image files of illustrations cropped out from original images
embeddings_file = args.embeddings_file  # path to CLIP embeddings
category = args.category  # visual category to isolate and crop
if not os.path.exists(cropped):
    os.makedirs(cropped)
# COMMENT OUT A STEP IF YOU DO NOT WISH TO RUN IT #
# PLEASE BE AWARE EACH STEP RELIES ON THE OUTPUTS OF THE PREVIOUS STEPS
crop_images.retrieve_cropped_images(labels_path, images_path, cropped, category)  # crop illustrations for CLIP
create_embs.make_embs(cropped, embeddings_file)  # create CLIP embeddings
