import logging

import numpy as np

import faceR.util.emb_helper as emb_helper
from faceR.util import openvino_helper


def embed(faces_gen, conf, device):
    """
    The same old embedding, but using NCS device for network forward pass
    :return: embeddings for a given list of faces
    """
    logger = logging.getLogger('openvino')
    logger.setLevel(logging.DEBUG)

    logger.info('using conf: ' + str(conf))
    model = conf[device]['model']
    plugin, exec_net, input_blob, out_blob = openvino_helper.load_network(model, device, conf)

    for frame, aligned_list in faces_gen:
        # batch = len(aligned_list)
        # new_faces = np.empty(shape=(batch, 3, 160, 160))
        faces = emb_helper.get_face_in_frame(frame, aligned_list, conf['image size'])
        emb_array = np.empty(shape=(len(aligned_list), 512))

        for i, face in enumerate(faces):
            face = np.transpose(face)
            for j in range(3):
                face[j] = np.transpose(face[j])
            # new_faces[i] = face.copy()
            res = exec_net.infer(inputs={input_blob: face})
            emb_array[i] = res['normalize']

        # new_faces = new_faces.reshape(-1, 3, 160, 160)
        # res = exec_net.infer(inputs={input_blob: new_faces})

        # emb_array = res['normalize']
        yield emb_array

    logger.info('exiting NCS2 facenet embedding')

    del exec_net
    del plugin
