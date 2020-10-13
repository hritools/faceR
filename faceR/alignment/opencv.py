import cv2
import logging


def align_faces(frames, conf):
    """
    Aligns the faces on image using haare cascades
    :return: frame and bboxes around faces
    """
    cascPath = conf['cascade path']
    faceCascade = cv2.CascadeClassifier(cascPath)

    for frame in frames:
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=conf['scale factor'],
            minNeighbors=conf['min neighbors'],
            minSize=(conf['min width'], conf['min height']),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        yield frame, [(x, y, x+w, y+h) for (x, y, w, h) in faces]

    logging.debug('exiting opencv align')
