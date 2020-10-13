"""
A stateful tracker of arbitrary objects
"""
import logging

import dlib
from multiprocessing import Pool

name_list = list()
d_trackers = list()


class ReusablePool:
    """
    Manage reusable tracker objects for use by client.
    """
    def __init__(self, size):
        logging.debug('Creating ' + str(size) + ' trackers!')
        self._reusables = [dlib.correlation_tracker() for _ in range(size)]

    def acquire(self, frame, bbox):
        if len(self._reusables) == 0:
            tracker = dlib.correlation_tracker()
            logging.debug('No spare trackers! Created new.')
        else:
            tracker = self._reusables.pop()
        tracker.start_track(frame, dlib.rectangle(
            bbox[0], bbox[1], bbox[2], bbox[3]))

        return tracker

    def release(self, reusable):
        self._reusables.append(reusable)


tracker_pool = ReusablePool(0)


def reset_trackers():
    """
    Resets trackers.
    """
    global d_trackers
    for tracker in d_trackers:
        tracker_pool.release(tracker)
    d_trackers = list()


def start_track(frame, face_pos):
    """
    Starts new trackers.
    :param frame: frame from which to start
    :param face_pos: positions of faces at the image
    """
    d_tracker = tracker_pool.acquire(frame, face_pos)
    d_trackers.append(d_tracker)


def update(d_tracker, frame):
    d_tracker.update(frame)
    return d_tracker.get_position()


def track(frame_gen):
    """
    Updates trackers.
    :param frame_gen: generator feeding frames
    """
    for frame in frame_gen:
        pos_list = list()
        for d_tracker in d_trackers:
            d_tracker.update(frame)
            pos = d_tracker.get_position()
            pos_list.append((int(pos.left()), int(pos.top()),
                             int(pos.right()), int(pos.bottom())))
        yield pos_list
    logging.debug('exiting dlib tracking')


# TODO: finalize multiprocessed tracking
def track_multiprocess(frame_gen, conf):
    """
    Updates trackers.
    :param conf: configs
    :param frame_gen: generator feeding frames
    """
    pool = Pool(conf['processes'])

    for frame in frame_gen:
        pos_list = list()

        # result = {
        #     d_tracker: pool.apply_async(lambda x: x.update(frame))
        #     for d_tracker in d_trackers
        # }
        results = [pool.apply_async(update, args=(d_tracker, frame)) for d_tracker in d_trackers]

        for res in results:
            pos = res.get()
            # d_tracker.update(frame)
            # pos = d_tracker.get_position()
            pos_list.append((int(pos.left()), int(pos.top()),
                             int(pos.right()), int(pos.bottom())))
        yield pos_list
    logging.debug('exiting dlib tracking')
