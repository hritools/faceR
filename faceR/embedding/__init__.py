from faceR.conf import get_entry
import logging


def embed(faces_gen):
    """
    Runs embedding for all the faces from faces_gen
    :param faces_gen: generator giving a pair of values (frame, list_of_aligned_faces) to embed
    :return: the list of embeddings for a given list of faces
    """
    emb = get_entry('recognition')['embedding']
    provider = emb['provider']
    framework = emb['framework']

    logger = logging.getLogger('embedding')
    logger.setLevel(logging.INFO)

    logger.info('embedding framework/provider/device: ' + emb['provider'])

    if framework == 'tf':
        from faceR.embedding import facenet_tf
        return facenet_tf.embed(faces_gen, emb[framework][provider])
    elif emb['framework'] == 'openvino':
        from faceR.embedding import facenet_openvino
        device = emb[framework]['device']
        return facenet_openvino.embed(faces_gen, emb[framework][provider], device)
