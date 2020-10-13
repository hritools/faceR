import logging
import pickle

import numpy as np


def classify(emb_gen, conf):
    """
    :param emb_gen: generator of faces to classify
    :param conf: configs for the classifier
    :return: a list of names for recognized faces
    """
    # restore the classifier
    # classifier, class_names = restore_classifier(conf['model'])
    with open(conf['model'], 'rb') as f:
        (le, svm) = pickle.load(f)

    for emb_array in emb_gen:
        # classify faces
        name_list = []

        predictions = svm.predict_proba(emb_array)
        maxI = [np.argmax(prediction) for prediction in predictions]
        people = [le.inverse_transform(maxi) for maxi in maxI]
        for i in range(len(people)):
            confidence = predictions[i][maxI[i]]
            if confidence > conf['threshold']:
                logging.debug("Predict {} with {:.2f} confidence.".format(people[i], confidence))
                name_list.append(people[i])
            else:
                logging.debug("Predict {} with {:.2f} confidence.".format(people[i], confidence))
                name_list.append('unknown')

        yield name_list

    logging.debug('exiting classifier')
