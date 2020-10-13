import logging
import time
from functools import reduce

from faceR import embedding, classification, tracking, alignment
from faceR.conf import get_entry


def combiner_filter(frame_gen):
    """
    An artificial filter added to combine all the other filters together.
    In short, it adapts all the pipelines needed to process image from camera
    into a single filter.
    Needed because the flow of data is not always the same: we need to align->embed->recognize
    faces only once in a while, in other frame we only track already recognized stuff.
    :param frame_gen: a generator from which to gen frames
    :return: a triple of values (frame, list of names whose faces were found on photo, list of bboxes around faces)
    """
    # empty generator to feed images to
    def image_feed():
        global image, running
        while running:
            yield image

    # empty generator to feed images and aligned faces
    def image_name_feed():
        global image, aligned_list, running
        while running:
            yield image, aligned_list

    global running
    running = True

    frame_interval = get_entry('recognition')['tracking']['frames']
    stamp = 0

    # here are those pipelines mentioned earlier: alignment, recognition, and tracking
    align_pipeline = [
        alignment.align_faces,
    ]
    rec_pipeline = [
        embedding.embed,
        classification.classify,
    ]
    track_pipeline = [
        tracking.track
    ]

    # collapsing pipelines to the last generator, so that when next()
    # is called on it, the control flow goes through the whole pipeline
    align_pipeline = reduce(lambda x, y: y(x), align_pipeline, image_feed())
    rec_pipeline = reduce(lambda x, y: y(x), rec_pipeline, image_name_feed())
    track_pipeline = reduce(lambda x, y: y(x), track_pipeline, image_feed())

    name_list = None
    logger = logging.getLogger('combiner')
    logger.setLevel(logging.DEBUG)

    checkpoints = [(int(time.time()*1000.0), 'started at')]
    for frame in frame_gen:
        global image, aligned_list
        image = frame
        checkpoints.append((int(time.time()*1000.0), 'getting frame / display'))

        if stamp % frame_interval == 0:
            frame, aligned_list = next(align_pipeline)

            checkpoints.append((int(time.time()*1000.0), 'align'))

            if len(aligned_list) > 0:
                name_list = next(rec_pipeline)
                checkpoints.append((int(time.time()*1000.0), 'recognition'))
            else:
                name_list = []

            tracking.reset_trackers()
            for face_pos in aligned_list:
                tracking.start_track(frame, face_pos)
        else:
            aligned_list = next(track_pipeline)
            checkpoints.append((int(time.time()*1000.0), 'tracking took '))

        stamp += 1

        logger.debug('%s ms\t whole pipe' % (int(time.time() * 1000 - checkpoints[0][0])))
        for i in range(1, len(checkpoints)):
            time_took = checkpoints[i][0] - checkpoints[i - 1][0]
            logger.debug("%s ms\t\t %s - checkpoint %s" % (int(time_took), checkpoints[i][1], i))
        checkpoints.clear()
        checkpoints.append((int(time.time()*1000.0), 'started at'))
        yield frame, name_list, aligned_list

    # make sure all the pipes terminate
    running = False
    logger.debug('making sure align pipe is clear')
    next(align_pipeline, None)
    logger.debug('making sure rec pipe is clear')
    next(rec_pipeline, None)
    logger.debug('making sure track pipe is clear')
    next(track_pipeline, None)

