from faceR.conf import get_entry

import logging
import faceR.classification.test_classifier
import faceR.classification.svm_classifier


def classify(emb_gen):
    """
    A generator, which gets a embedding list generator as an input.

    :param emb_gen: a generator of embeddings to recognize
    :return: a list of names in the exact order of input faces
    """
    classif = get_entry('recognition')['classification']

    logger = logging.getLogger('classification')
    logger.setLevel(logging.INFO)

    logger.info('classification provider: ' + classif['provider'])

    if classif['provider'] == 'test-classifier':
        return test_classifier.classify(emb_gen, classif['test-classifier conf'])
    elif classif['provider'] == 'svm-classifier':
        return svm_classifier.classify(emb_gen, classif['svm-classifier conf'])
