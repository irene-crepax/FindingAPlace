# PIPELINE FILE FOR LAYOUT PARSING AND OCR #
# Import functions from modules #
from ocr import ocr
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str,
                    help='path to original images')
parser.add_argument('--labels_path', type=str,
                    help='path to folder to store layout files')
parser.add_argument('--captions_output', type=str,
                    help='csv file to store ocr results')
parser.add_argument('--ocr_categories', nargs='+', default=['caption'],
                    help='ocr_categories')
args = parser.parse_args()
images_path = args.images_path  # path to folder of images to parse
labels_path = args.labels_path  # path to folder to store layout files
captions_output = args.captions_output  # csv file to store ocr
ocr_categories = args.ocr_categories

# COMMENT OUT A STEP IF YOU DO NOT WISH TO RUN IT #
# PLEASE BE AWARE EACH STEP RELIES ON THE OUTPUTS OF THE PREVIOUS STEPS
ocr.bruteforceocr(labels_path, images_path, ocr_categories, captions_output)  # run ocr


