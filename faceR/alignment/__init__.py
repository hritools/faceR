from faceR.conf import get_entry

import logging


def align_faces(frames):
    """
    A generator, which gets a frame generator as an input.

    :param frames: a generator of frames in which to align faces
    :return: a frame itself, and a list of bboxes around the faces.
    """
    align = get_entry('recognition')['face alignment']
    provider = align['provider']

    logger = logging.getLogger('alignment')
    logger.setLevel(logging.INFO)

    logger.info('alignment provider: ' + provider)
    if provider == 'opencv':
        from faceR.alignment import opencv as aligner
    elif provider == 'dlib':
        import dlib_al as aligner
    elif provider == 'mtcnn':
        from faceR.alignment import mtcnn as aligner
    elif provider == 'face-boxes':
        from faceR.alignment import face_boxes as aligner
    elif provider == 'adas':
        from faceR.alignment import adas as aligner

    return aligner.align_faces(frames, align[provider])
