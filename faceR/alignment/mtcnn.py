import logging
import numpy as np
import tensorflow as tf

from faceR.FaceNet import detect_face, facenet


def restore_mtcnn_model(conf):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, conf['model folder'])
    return pnet, rnet, onet


def align_faces(frames, conf):
    minsize = conf['minsize']
    pnet, rnet, onet = restore_mtcnn_model(conf)

    for frame in frames:
        threshold = conf['threshold']  # three steps's threshold
        factor = conf['factor']  # scale factor

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]

        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        # num_of_faces = bounding_boxes.shape[0]
        aligned_list = []
        for face_pos in bounding_boxes:
            face_pos = np.rint(face_pos)
            aligned_list.append(face_pos.astype(int))

        yield frame, aligned_list

    logging.debug('exiting mtcnn align')
