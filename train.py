#!/usr/bin/env python3

""" Training Script """

import os
from fastai.vision.all import *
from duckduckgo_search import DDGS


def search_images(term, max_images=64):
    """ Search for images to train on """
    print(f'Searching for "{term}"')
    return L(DDGS().images(term, max_results=max_images)).itemgot('image')


def download_and_resize_images(search_terms, path='static/images'):
    """
    Downloads images for each search term and resizes them for training.

    Arguments:
    - search_terms: list of categories to search for.
    - path: destination directory where images will be saved.
    """
    path = Path(path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    for search_term in search_terms:
        dest = (path/search_term)
        dest.mkdir(parents=True, exist_ok=True)
        download_images(dest, urls=search_images(f'{search_term} photo',
                        max_images=64))
        time.sleep(5)
        resize_images(path/search_term, max_size=400, dest=path/search_term)

    failed = verify_images(get_image_files(path))
    if failed:
        print(f"Removing {len(failed)} failed images.")
        failed.map(Path.unlink)


def train_classifier(categories):
    """ Train a classifier on a list of categories """
    path = Path('static/images')
    download_and_resize_images(categories, path)
    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2,
                                       seed=42,
                                       item_tfms=Resize((224, 224)))

    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(5)
    model_save_path = '../models/export.pkl'
    learn.export(model_save_path)
    print('Training complete!')
    print(f'Model saved to {model_save_path}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Error: Please provide at least 2 categories for training.")
        sys.exit(1)

    # Get categories from command-line arguments
    categories = sys.argv[1:]
    # categories = ['forest', 'bird']

    print(f"Training on categories: {categories}")
    train_classifier(categories)
