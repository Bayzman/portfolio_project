#!/usr/bin/env python3

""" Training Script """

import os
import shutil
import zipfile
from pathlib import Path
from fastai.vision.all import *
from duckduckgo_search import DDGS
import time


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


def extract_zip_data(zip_path, extract_to='static/images'):
    """ Extract the uploaded zip file to the given directory """
    print(f'Extracting {zip_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Ensure that there are at least two subdirectories (for different classes)
    extract_dir = Path(extract_to)
    subfolders = [f for f in extract_dir.iterdir() if f.is_dir()]
    if len(subfolders) < 2:
        raise ValueError("The zip file must contain at least two subfolders representing different classes.")
    
    print(f'Extracted data from {zip_path}')


def train_classifier(categories, data_source=None):
    """ Train a classifier on a list of categories """
    path = Path('static/images')

    if data_source:
        # If a zip file is provided, extract it
        extract_zip_data(data_source, path)
    else:
        # Otherwise, download and resize images based on search terms
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

    # Check if categories or a file path is provided
    if len(sys.argv) < 2:
        print("Error: Please provide at least 2 categories for training or a path to the zip file containing data.")
        sys.exit(1)

    # If the second argument is a file, it is the path to the zip file
    data_source = None
    if os.path.isfile(sys.argv[1]):
        data_source = sys.argv[1]
        print(f"Using provided zip file: {data_source}")
        categories = []
    else:
        # Get categories from command-line arguments (if no file is provided)
        categories = sys.argv[1:]
        print(f"Training on categories: {categories}")

    # Start training based on provided data (either from zip or search terms)
    train_classifier(categories, data_source)
