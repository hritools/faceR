from faceR.conf import get_entry

import logging
import faceR.display.opencv
import faceR.display.dummy


def show(final_gen):
    """
    A consumer for a generator, feeding in three variables at each step:
    1. frame,
    2. list of aligned faces,
    3. list of names on photo.
    Shows a window with the final result
    """
    logger = logging.getLogger('display')
    logger.setLevel(logging.INFO)

    emb = get_entry('recognition')['embedding']
    provider = emb['provider']
    framework = emb['framework']

    logger.info('display provider: ' + get_entry('display')['provider'])

    if get_entry('display')['provider'] == 'opencv':
        return opencv.show(final_gen, emb[framework][provider]['image size'])
    if get_entry('display')['provider'] == 'dummy':
        return dummy.show(final_gen, emb[framework][provider]['image size'])
