# PIPELINE FILE FOR LAYOUT PARSING AND OCR #
# Import functions from modules #
from layout_parsing import prepare_data_for_LP
from layout_parsing import parse_layout
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--annotations_file', type=str,
                    help='temporary json file')
parser.add_argument('--images_path', type=str,
                    help='path to original images')
parser.add_argument('--labels_path', type=str,
                    help='path to folder to store layout files')
parser.add_argument('--dataset_name', type=str,
                    help='chosen name of dataset')
parser.add_argument('--output_path', type=str, default=False,
                    help='path to store images with bounding boxes; leave as default if you do not wish to save a '
                         'copy of the images')
parser.add_argument('--model_path', type=str, default='model_final.pth',
                    help='path to Layout Parser model weights')
parser.add_argument('--categories', nargs='+', default=['caption', 'text', 'figure', 'title'],
                    help='categories')
args = parser.parse_args()
annotations_file = args.annotations_file
images_path = args.images_path  # path to folder of images to parse
labels_path = args.labels_path  # path to folder to store layout files
dataset_name = args.dataset_name  # chosen name of dataset
output_path = args.output_path  # path to folder to store images with bounding boxes; change to False if you do not wish to save the images
model_path = args.model_path  # path to Layout Parser fine-tuned model
search_term = args.search_term
cats = args.categories
# folders creation #
if not os.path.exists(labels_path):   #### fix these to skip if those args are not provided
    os.makedirs(labels_path)
if output_path != False and not os.path.exists(output_path):
    os.makedirs(output_path)

# COMMENT OUT A STEP IF YOU DO NOT WISH TO RUN IT #
# PLEASE BE AWARE EACH STEP RELIES ON THE OUTPUTS OF THE PREVIOUS STEPS
prepare_data_for_LP.create_empty_annotations(images_path, annotations_file)  # creates empty annotations for Layout Parser step
parse_layout.predict(annotations_file, images_path, labels_path, dataset_name, model_path, cats, output_path)  # run Layout Parser step


