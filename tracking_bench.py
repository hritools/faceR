import time
import cv2
from alignment import dlib_al
import embedding.identify_face as identify_face
import tensorflow as tf


def redraw_frame(image, aligned):
    i = 0
    j = 0
    for faces, method in aligned:
        for face in faces:
            cv2.rectangle(image, (face[0], face[1]),
                          (face[2], face[3]), (255 if j == 0 else 0, 255 if j == 1 else 0, 255 if j == 2 else 0), 2)
            cv2.putText(image, method, (face[0], face[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255 if j == 0 else 0, 255 if j == 1 else 0, 255 if j == 2 else 0), thickness=2, lineType=2)

            i += 1
            # if names[i] == 'others':
            #     cv2.putText(image, names[i], (face[0], face[1] - 30),
            #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2, lineType=2)
            #     cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
            # else:
            #     cv2.putText(image, names[i], (face[0], face[1] - 30),
            #                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2, lineType=2)

        j += 1


detector = dlib_al.get_frontal_face_detector()

__camera__ = cv2.VideoCapture(0)
dets = None

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 50
config.inter_op_parallelism_threads = 5
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

with tf.Graph().as_default() and tf.Session(config=config) as sess:
    # restore mtcnn model
    pnet, rnet, onet = identify_face.restore_mtcnn_model()

    while __camera__.isOpened():
        aligned_list = list()
        ret, img = __camera__.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        starting_time = time.time()
        # img = dlib.load_rgb_image(frame)
        aligned_list.append((identify_face.align_mtcnn(img, pnet, rnet, onet, 40), 'mtcnn'))
        # print(aligned_list[0][0])
        print(len(aligned_list[0][0]))
        print('mtcnn face alignment: \t' + str((time.time() - starting_time) * 1000) + 'ms: \t' + str(len(aligned_list[0][0])) + ' faces')

        starting_time = time.time()
        dets = detector(img)
        aligned_list.append(([(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())) for pos in dets], 'dlib'))
        print('dlib face detector: \t' + str((time.time() - starting_time) * 1000) + 'ms: \t' + str(len(aligned_list[1][0])) + ' faces')

        starting_time = time.time()
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Create the haar cascade
        aligned_list.append(([(x, y, x+w, y+h) for (x, y, w, h) in faces], 'haare'))
        print('haare cascade:  \t\t' + str((time.time() - starting_time) * 1000) + 'ms: \t' + str(len(aligned_list[2][0])) + ' faces')

        # print("Number of faces detected: {}".format(len(dets)))

        names_list = ['me' for d in aligned_list]

        redraw_frame(img, aligned_list)

        cv2.imshow("stream", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

cv2.destroyAllWindows()
