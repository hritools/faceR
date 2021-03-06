import os
import time
import uuid

import cv2
import logging

import imageio

from faceR.util import emb_helper
from faceR.util.image_helper import redraw_frame


def show(final_feed, image_size):
    """
    Simply consumes the feed, showing the final result using opencv
    """
    logger = logging.getLogger('display')
    logger.setLevel(logging.DEBUG)
    logger.debug('using opencv for displaying')
    names_write = []

    for frame, name_list, aligned_list in final_feed:
        start = int(time.time() * 1000.0)

        if name_list and list(set(name_list) - set(names_write)):
            new_faces = list(set(name_list) - set(names_write))
            print('new faces found: ' + str(new_faces))

            names_write = names_write + new_faces
            with open('faces/faces_found.txt', 'w') as f:
                for item in names_write:
                    f.write("%s\n" % item)
            for name in names_write:
                if not os.path.exists('faces/cap/%s' % name):
                    os.makedirs('faces/cap/%s' % name)
        faces = emb_helper.get_face_in_frame(frame, aligned_list, image_size)
        for i in range(len(faces)):
            imageio.imwrite('faces/cap/%s/%s.jpg' % (name_list[i], str(uuid.uuid4())), faces[i])
        logger.debug('%s ms\t\t saving faces took' % (int(time.time() * 1000.0) - start))

        frame = redraw_frame(frame, name_list, aligned_list)
        cv2.imshow("stream", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        logger.debug('%s ms\t\t displaying took' % (int(time.time() * 1000.0) - start))

    logger.info('destroying display frame')

    cv2.destroyAllWindows()
