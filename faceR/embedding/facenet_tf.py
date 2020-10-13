import logging
import tensorflow as tf
import numpy as np

from faceR.FaceNet import facenet
import faceR.util.emb_helper as emb_helper


def embed(faces_gen, conf):
    """
    :return: embeddings for a given list of faces
    """
    logger = logging.getLogger('tf embed')
    logger.setLevel(logging.DEBUG)

    logger.info('using conf: ' + str(conf))

    gpu_options = tf.GPUOptions(allow_growth=True)

    g = tf.Graph()
    with g.as_default(), \
            tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False), graph=g) as sess:
        facenet.load_model(conf['model'])
        # get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        for frame, aligned_list in faces_gen:
            faces = emb_helper.get_face_in_frame(frame, aligned_list, conf['image size'])
            emb_array = np.zeros((len(faces), embedding_size))

            # run forward pass to calculate embeddings
            feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            yield emb_array

    logger.info('exiting facenet embedder')
