import logging
import sys

import numpy as np
from mvnc import mvncapi

from faceR.util import emb_helper


def embed(faces_gen, conf):
    """
    The same old embedding, but using NCS device for network forward pass
    :return: embeddings for a given list of faces
    """
    logging.debug('using conf: ' + str(conf))
    # The graph file that was created with the ncsdk compiler
    graph_file_name = conf['graph file']

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = mvncapi.Graph('graph1')

    devices = mvncapi.enumerate_devices()
    device_to_use = conf['device to use']
    if len(devices) < (device_to_use + 1):
        sys.exit("NCS device " + str(device_to_use) + " can't be found!")
    device = mvncapi.Device(devices[device_to_use])
    device.open()

    input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_in_memory)
    for frame, aligned_list in faces_gen:
        faces = emb_helper.get_face_in_frame(frame, aligned_list, conf['image size'])
        emb_array = np.empty(shape=(len(aligned_list), 512))

        for i, face in enumerate(faces):
            # Write the image to the input queue and queue the inference in one call
            graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, face.astype(np.float32), None)

            # Get the results from the output queue
            output, user_obj = output_fifo.read_elem()
            emb_array[i] = output

        yield emb_array

    logging.debug('exiting NCS facenet embedding')

    logging.debug('destroying graph')
    graph.destroy()
    logging.debug('closing NSC queues')
    input_fifo.destroy()
    output_fifo.destroy()
    logging.debug('destroying device')
    device.close()
    device.destroy()
