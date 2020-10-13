"""A basic frame receiver, should be used in case video must be transmitted"""

import cv2
import zmq
import base64
import numpy as np
import logging


def get_image(camera):
    """
    A generator feeding frames of the video
    """
    context = zmq.Context()
    footage_socket = context.socket(zmq.SUB)
    logging.debug('using image stream')
    logging.debug('listening to: tcp://*:' + camera['receiver']['port'])
    # listen the port specified in the configs
    footage_socket.bind('tcp://*:' + camera['receiver']['port'])
    footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
    logging.debug('connection acquired!')

    while True:
        try:
            frame = footage_socket.recv_string()

            # decode the image from text format
            img = base64.b64decode(frame)
            npimg = np.fromstring(img, dtype=np.uint8)
            source = cv2.imdecode(npimg, 1)
            yield source

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break
