import dlib
import logging


def align_faces(frames, conf):
    """
    Aligns the faces on image using dlib's built-in function
    :return: frame and bboxes around faces
    """
    detector = dlib.get_frontal_face_detector()
    logging.debug('using conf: ' + str(conf))

    for frame in frames:
        dets = detector(frame)
        yield frame, [(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())) for pos in dets]

    logging.debug('exiting dlib align')
