import time
import cv2
import logging
import signal
from threading import Thread, Lock, Condition
import cv2
import numpy


def exit_gracefully(sig, frame):
    global running
    running = False
    logger.info('Ctrl+C detected: exit procedure commenced!')


class WebcamVideoStream:
    def __init__(self, logger, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.should_capture_frame = Condition()
        self.logger = logger
        self.n = 0

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:

            self.n += 1

            logger.debug('taking a shot! %s' % self.n)
            self.frame = cv2.cvtColor(self.stream.read()[1], cv2.COLOR_BGR2RGB)
            logger.debug('took a shot! %s' % self.n)
            with self.should_capture_frame:
                self.should_capture_frame.wait()

    def read(self):
        to_return = numpy.copy(self.frame)
        with self.should_capture_frame:
            self.should_capture_frame.notifyAll()
        logger.debug('returning a frame %s' % self.n)

        return to_return

    def stop(self):
        self.stopped = True


def get_image(camera_conf):
    """
    Gets image from the default /dev/video0 device
    :return: generated frame
    """
    global running
    global logger
    logger = logging.getLogger('camera')
    running = True
    signal.signal(signal.SIGINT, exit_gracefully)

    width = camera_conf['width']
    height = camera_conf['height']
    device = camera_conf['video device']
    interval = 1 / camera_conf['framerate']

    vs = WebcamVideoStream(logger, int(device))
    vs.start()
    # infer interval between frames from the framerate
    logger.setLevel(logging.DEBUG)

    logger.info('using built-in camera')
    # logger.info(cv2.getBuildInformation())
    logger.info('interval between frames: ' + str(interval) + 'seconds')

    end_time = 0
    while running:
        time_took = time.time()
        frame = vs.read()
        frame = cv2.resize(frame, (width, height))

        cur_time = time.time()
        if (interval - (cur_time - end_time)) > 0:
            logger.debug('Sleeping for: ' + str((interval - (time.time() - end_time)) * 1000.0) + ' ms')
            time.sleep(interval - (cur_time - end_time))

        time_took = int((time.time() - time_took) * 1000)
        logger.debug('%s ms\t taking shot' % time_took)
        end_time = time.time()
        yield frame
    vs.stop()
