# !/usr/bin/env python
from functools import reduce

import argparse
import os
import unittest
import imageio

from faceR import camera
from faceR import combiner
from faceR.classification import test_classifier
from faceR.data.dataset import DataSet
from faceR.alignment import align_faces
from faceR.display import show
from faceR.conf import setup, get_entry
from faceR.util import emb_helper

name = "facer"


def init(config=None, logging=None):
    if not config:
        config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not logging:
        logging = os.path.join(os.path.dirname(__file__), 'logging.yaml')

    setup(config_filename=config, logging_config=logging)


def run():
    # now launching the whole thing boils down to simply combining the
    # following pipeline: camera->combiner->display
    pipeline = [
        combiner.combiner_filter,
        show
    ]
    reduce(lambda x, y: y(x), pipeline, camera.get_image())


def align(raw_images_dir, save_dir):
    global image
    emb = get_entry('recognition')['embedding']
    provider = emb['provider']
    framework = emb['framework']

    im_size = emb[framework][provider]['image size']

    def image_gen():

        while True:
            yield image

    aligner = align_faces(image_gen())
    ds = DataSet(raw_images_dir, im_size)
    for person in ds.people:
        person_dir = person.directory.replace(raw_images_dir, save_dir, 1)
        for pic in person.pics:
            image = imageio.imread(pic.path)
            frame, aligned = next(aligner)
            if len(aligned) != 1:
                print('Error! Expected to find 1 face in %s, but found %d' % (pic.path, len(aligned)))
            else:
                image_bin = emb_helper.get_face_in_frame(frame, aligned, im_size)[0]
                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)
                imageio.imwrite(os.path.join(person_dir, os.path.basename(pic.path)), image_bin)


def cluster(dataset_dir):
    pass


def train(aligned_images_dir, classifier_name):
    emb = get_entry('recognition')['embedding']
    provider = emb['provider']
    framework = emb['framework']

    im_size = emb[framework][provider]['image size']

    print('Loading dataset')
    dataset = DataSet(aligned_images_dir, im_size)
    print('Calculating embeddings')
    dataset.embed()
    print('Training classifier')
    test_classifier.train(dataset, classifier_name)


def test():
    # import faceR.alignment.test.test_al
    loader = unittest.TestLoader()
    base_dir = os.path.dirname(__file__)
    suite = loader.discover(base_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)


def run_cmd(argv):
    parser = argparse.ArgumentParser(description='Usually run for inference, can also be used to '
                                                 'train classifier and cluster faces.')
    parser.add_argument('task', type=str, choices=['run', 'align', 'cluster', 'train_classifier', 'test'],
                        help='Indicates what task to perform: '
                             '\'run\'  - process video feed; '
                             '\'align\'  - align faces on images and save algined faces '
                             '(requires raw_images_dir, save_dir); '
                             '\'cluster\'  - cluster faces in the directory (requires dataset_dir); '
                             '\'train_classifier\'  - train a classifier '
                             '(requires aligned_dir, classifier_name), replaces previous classifier file.'
                             '\'test\'  - run tests ')
    parser.add_argument('--raw_images_dir', type=str, required=False, help='Directory from where to get images.')
    parser.add_argument('--save_dir', type=str, required=False, help='Where to save aligned faces')
    parser.add_argument('--dataset_dir', type=str, required=False, help='Directory where to get faces for alignment')
    parser.add_argument('--aligned_dir', type=str, required=False, help='Directory from where to get images to train '
                                                                        'classifier')
    parser.add_argument('--classifier_name', type=str, required=False, help='How to name classifier')

    parser.add_argument('--config', type=str, required=False,
                        help='Configuration file in yaml format, default is located at faceR/config.yaml.')
    parser.add_argument('--logging', type=str, required=False,
                        help='Logging config in yaml, default is located at logging.yaml')

    arguments = parser.parse_args(argv)
    config = arguments.config if arguments.config else 'config.yaml'
    logging = arguments.logging if arguments.logging else 'logging.yaml'
    init(config, logging)
    if arguments.task == 'run':
        run()
    if arguments.task == 'align':
        if not arguments.raw_images_dir or not arguments.save_dir:
            parser.error("\'align\' requires \'raw_images_dir\' and \'save_dir\'.")
        else:
            align(arguments.raw_images_dir, arguments.save_dir)
    if arguments.task == 'cluster':
        if not arguments.dataset_dir:
            parser.error("\'cluster\' requires \'dataset_dir\'.")
        else:
            cluster(arguments.dataset_dir)
    if arguments.task == 'train_classifier':
        if not arguments.aligned_dir or not arguments.classifier_name:
            parser.error("\'train_classifier\' requires \'aligned_dir\' and \'classifier_name\'.")
        else:
            train(arguments.aligned_dir, arguments.classifier_name)
    if arguments.task == 'test':
        test()
