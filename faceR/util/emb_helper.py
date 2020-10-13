import cv2
import numpy as np


def get_face_in_frame(frame, aligned_list, img_size=160):
    images = np.zeros((len(aligned_list), img_size, img_size, 3))
    i = 0
    for face_pos in aligned_list:
        face = [max(face_pos[0], 0), max(face_pos[1], 0), min(face_pos[2], frame.shape[1]),
                min(face_pos[3], frame.shape[0])]

        img = frame[face[1]:face[3], face[0]:face[2], ]
        if img.ndim == 2:
            img = to_rgb(img)
        img = cv2.resize(img, (img_size, img_size))

        # img = misc.imresize(img, (img_size, img_size), interp='bilinear')
        img = prewhiten(img)
        img = crop(img, False, img_size)
        images[i, :, :, :] = img
        i += 1
    return images


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image
