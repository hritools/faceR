from faceR.conf import get_entry
from . import opencv_capture
from .streaming import receiver
import logging


def get_image():
    """
    Image feed adapter
    :return: frame generator, which one exactly depends on the config file
    """
    camera = get_entry('camera')
    logger = logging.getLogger('camera')
    logger.setLevel(logging.INFO)

    logger.info('camera settings: ' + str(camera))
    if camera['provider'] == 'opencv':
        return opencv_capture.get_image(camera)
    elif camera['provider'] == 'text-stream':
        return receiver.get_image(camera)
