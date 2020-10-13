import logging

from faceR.conf import get_entry
from . import dlib_tracking

name_list = list()

# As long as there are no other candidates at the moment for tracking,
# d_lib's implementation of tracking is currently used and no parameters for tracking
start_track = dlib_tracking.start_track
reset_trackers = dlib_tracking.reset_trackers


def track(faces_gen):
    """
    Runs embedding for all the faces from faces_gen
    :param faces_gen: generator giving a pair of values (frame, list_of_aligned_faces) to embed
    :return: the list of embeddings for a given list of faces
    """
    track_conf = get_entry('recognition')['tracking']

    logging.debug('tracking provider: ' + track_conf['provider'])
    if track_conf['provider'] == 'dlib' and track_conf['dlib conf']['multiprocessing']:
        return dlib_tracking.track_multiprocess(faces_gen, track_conf['dlib conf'])
    elif track_conf['provider'] == 'dlib' and not track_conf['dlib conf']['multiprocessing']:
        return dlib_tracking.track(faces_gen)
