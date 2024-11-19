# PIPELINE FILE FOR LAYOUT PARSING AND OCR #
# Import functions from modules #
from training import train_model
from training import convert_annotations
from training import split_annotations
from training import split_directory
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str,
                    help='path to annotated images')
parser.add_argument('--categories', nargs='+', default=['caption', 'text', 'figure', 'title'],
                    help='categories')
parser.add_argument('--name', type=str,
                    help='name of your dataset')
parser.add_argument('--folders', nargs='+',
                    help='training and validation folders')
args = parser.parse_args()
file_path = args.file_path
categories = args.categories
name = args.name
folders = args.folders

convert_annotations.convert(file_path, categories)
split_annotations.split(categories)
train_model.run_training(file_path, name, categories)
split_directory.split_dir(folders)
