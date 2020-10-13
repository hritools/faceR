"""Frame sender, should be used in case video must be transmitted"""

import base64
import time

import cv2
import zmq
import logging
import sys

from faceR.conf import get_entry


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# set up the networking
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
cam_conf = get_entry('camera')
logging.debug('tcp://' + cam_conf['receiver']['ip'] + ':' + cam_conf['receiver']['port'])
footage_socket.connect('tcp://' + cam_conf['receiver']['ip'] + ':' + cam_conf['receiver']['port'])

# infer interval between frames from the framerate
interval = 1 / cam_conf['framerate']
logging.debug('interval between frames: ' + str(interval) + 'seconds')

__camera__ = cv2.VideoCapture(cam_conf['video device'])
width = cam_conf['width']
height = cam_conf['height']

while True:
    try:
        starting_time = time.time()
        grabbed, frame = __camera__.read()          # grab the current frame
        frame = cv2.resize(frame, (width, height))   # resize the frame
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)

        if interval - (time.time() - starting_time) > 0:
            logging.debug('sleeping for: ' + str(interval - (time.time() - starting_time)) + ' seconds')
            time.sleep(interval - (time.time() - starting_time))

    except KeyboardInterrupt:
        __camera__.release()
        break
