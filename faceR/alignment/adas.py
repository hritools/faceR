import logging

import cv2
import numpy as np

from faceR.util import openvino_helper as helper


def align_faces(frames, conf):
    device = conf['device']
    #  Load network
    plugin, exec_net, input_blob, out_blob = helper.load_network(conf[device]['model'], device, conf)
    height = conf['height']
    width = conf['width']
    threshold = conf['threshold']

    for frame in frames:
        r_frame = cv2.resize(frame, (width, height))
        f_h = frame.shape[0]
        f_w = frame.shape[1]

        r_frame = [r_frame[:, :, 0], r_frame[:, :, 1], r_frame[:, :, 2]]
        dets = exec_net.infer(inputs={input_blob: r_frame})['detection_out'][0][0]

        i = 0
        faces = np.empty([0, 4], dtype=int)
        while dets[i][2] > threshold:
            faces = np.insert(faces,
                              i,
                              np.array(
                                  [dets[i][3] * f_w,
                                   dets[i][4] * f_h,
                                   dets[i][5] * f_w,
                                   dets[i][6] * f_h]).astype(int),
                              axis=0)
            i += 1

        yield frame, faces

    logger = logging.getLogger('face det')
    logger.debug('exiting adas face detection')
    del exec_net
    del plugin
