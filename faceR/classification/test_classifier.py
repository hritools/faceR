import logging
import os
import pickle
import numpy as np
from sklearn.svm import SVC


def restore_classifier(classifier_path):
    # load the classifier
    logging.debug('Loading face classifier')
    if os.path.isdir(classifier_path):
        classifier = []
        class_names = []
        for root, dirs, files in os.walk(classifier_path):
            for file in files:
                with open(os.path.join(root, file), 'rb') as infile:
                    tmp1, tmp2 = pickle.load(infile)
                    classifier.append(tmp1)
                    class_names.append(tmp2)

    if os.path.isfile(classifier_path):
        with open(classifier_path, 'rb') as infile:
            classifier, class_names = pickle.load(infile)

    return classifier, class_names


def train(dataset, classifier_filename):
    classifier_filename = os.path.join('models', 'classifier', classifier_filename) + '.pkl'
    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    dataset.embed()

    images = dataset.get_all_images()
    emb_array = [pic.embedding for pic in images]
    labels = [pic.person for pic in images]

    # Train classifier
    print('Training classifier')

    model = SVC(kernel='linear', probability=True)
    model.fit(emb_array, labels)

    # Create a list of class names
    class_names = [person.name.replace('_', ' ') for person in dataset.people]

    # Saving classifier model
    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((model, class_names), outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved classifier model to file "%s"' % classifier_filename)


def classify(emb_gen, conf):
    """
    :param emb_gen: generator of faces to classify
    :param conf: configs for the classifier
    :return: a list of names for recognized faces
    """
    # restore the classifier
    classifier, class_names = restore_classifier(conf['model'])

    for emb_array in emb_gen:
        # classify faces
        predict = classifier.predict_proba(emb_array)
        best_class_indices = np.argmax(predict, axis=1)
        best_class_prob = predict[np.arange(len(best_class_indices)), best_class_indices]

        name_list = []
        for i in range(len(best_class_indices)):
            if class_names[best_class_indices[i]] == 'others' or best_class_prob[i] < conf['threshold']:
                logging.debug('<ERROR> %s: %.3f' % (class_names[best_class_indices[i]], best_class_prob[i]))
                name_list.append('unknown')
            else:
                logging.debug('%s: %.3f' % (class_names[best_class_indices[i]], best_class_prob[i]))
                name_list.append(class_names[best_class_indices[i]])

        yield name_list
    logging.debug('exiting classifier')
