# PIPELINE FOR CLIP SEARCH #

from search_dataset import caption_search
from search_dataset import clip_search
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_file', type=str, default='',
                    help='file to CLIP embeddings')
parser.add_argument('--search_prompt', default=[], nargs='+',
                    help='search prompt: choose either an image for image-to-image search or type a sentence for '
                         'text-to-image search')
parser.add_argument('--type_of_search', type=str, default='text',
                    help='if searching through captions, please type caption; if running image-to-text, please type '
                         'image; if running text-to-image, leave as default')
parser.add_argument('--found_images', type=str,
                    help='csv file to save images found')
parser.add_argument('--captions_file', type=str, default='',
                    help='csv file of captions')
parser.add_argument('--search_term', type=str, default='',
                    help='search term for captions')
args = parser.parse_args()
t = args.search_prompt  # choose an image for CLIP image-to-image search or a text prompt fpr CLIP text-to-image search
embeddings_file = args.embeddings_file  # path to CLIP embeddings
i = args.type_of_search  # change to image if t is an image file
found_images = args.found_images  # file to save image found through CLIP search
captions_file = args.captions_file  # csv file of ocr'd captions
search_term = args.search_term  # term to search the captions for

if i == 'text' or i == 'image':
    clip_search.clip_search(t, embeddings_file, i, found_images)  # run CLIP search
else:
    caption_search.search_captions(search_term, captions_file, found_images)

