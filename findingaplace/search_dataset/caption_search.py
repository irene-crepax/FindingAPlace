import pandas as pd
from thefuzz import fuzz, process
from nltk.util import ngrams
import string


def search_captions(query, captions_file, found_images):
    all_captions = pd.read_csv(captions_file)
    columns = all_captions.loc[:,
              all_captions.columns != 'Page'].columns.tolist()
    columns = [item for item in columns if item.startswith('text')]

    n = len(query.split())

    captions = list()
    all_scores = list()
    j = 0
    for r, row in all_captions.iterrows():
        for c in columns:
            if isinstance(row[c], str):
                caption = row[c]
                captions.append(caption.lower())
                caption = caption.translate(str.maketrans('', '', string.punctuation))
                n_grams = list()
                for i in range(n):
                    n_grams.extend(list(ngrams(caption.split(), i + 1)))
                n_grams = [' '.join(tup) for tup in n_grams]
                for n_gram in n_grams:
                    all_scores.append({'image': str(r), 'sentence': str(j), 'ngram': n_gram,
                                       'score': (fuzz.token_sort_ratio(n_gram, query))})
                j += 1

    new_scores = sorted(all_scores, key=lambda d: d['score'], reverse=True)
    new_scores = sorted(new_scores, key=lambda d: d['image'])
    all_scores = [new_scores[0]]
    for item in new_scores:
        if item['image'] != all_scores[len(all_scores)-1]['image']:
            all_scores.append(item)
    all_scores = sorted(all_scores, key=lambda d: d['score'], reverse=True)
    sorted_images = list()
    for score in all_scores:
        sorted_images.append(int(score['image']))
    filtered_df = all_captions.reindex(sorted_images)
    filtered_df.to_csv(found_images, index=False)
