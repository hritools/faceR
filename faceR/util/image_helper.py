import threading

import numpy as np
from PIL import ImageFont, ImageDraw, Image


class ThreadsafeIter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadsafeIter(f(*a, **kw))

    return g


def redraw_frame(image, names, aligned):
    """
    Adds names and bounding boxes to the frame
    """
    i = 0
    unicode_font = ImageFont.truetype("DejaVuSansMono.ttf", size=17)

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    for face in aligned:
        draw.rectangle((face[0], face[1], face[2], face[3]), outline=(0, 255, 0), width=2)

        if names is not None and len(names) > i:

            if names[i] == 'unknown':
                draw.text((face[0], face[1] - 30), "unknown", fill=(0, 0, 255), font=unicode_font)
                draw.rectangle((face[0], face[1], face[2], face[3]), outline=(0, 0, 255), width=2)
            else:
                draw.text((face[0], face[1] - 30), names[i], fill=(0, 255, 0), font=unicode_font)
            if names is None or len(names) <= i:
                draw.text((face[0], face[1] - 30), 'refreshing...', fill=(255, 0, 0), font=unicode_font)
            i += 1
    return np.array(img_pil)
