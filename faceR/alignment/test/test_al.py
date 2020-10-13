import os
import unittest

import cv2
import yaml

from faceR.alignment.dlib_al import align_faces as dlib_align
from faceR.alignment.opencv import align_faces as opencv_align
from faceR.alignment.mtcnn import align_faces as mtcnn_align
from faceR.util.image_helper import redraw_frame


def load_faces(faces_im):
    faces = []
    with open(faces_im) as f:
        content = f.read().splitlines()

    for line in content:
        img_name = os.path.join(os.path.dirname(__file__), line.split(' ')[0])
        bbox = line.split(' ')[1:]
        bbox = list(map(lambda x: int(x), bbox))
        img = cv2.imread(img_name)

        faces.append((img, bbox, img_name))
        face_al = img.copy()
        face_al = redraw_frame(face_al, ['given'], [bbox])
        cv2.imwrite(img_name.replace('raw', 'faces_input'), face_al)
    return faces


def save_face(face, bbox, dir):
    face_al = face.copy()
    face_al = redraw_frame(face_al, ['given'], bbox)
    cv2.imwrite(dir, face_al)


def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def boxes_common(bbox1, bbox2):
    common = list()
    common.append(max(bbox1[0], bbox2[0]))
    common.append(max(bbox1[1], bbox2[1]))
    common.append(min(bbox1[2], bbox2[2]))
    common.append(min(bbox1[3], bbox2[3]))

    return 2 * area(common) / (area(bbox1) + area(bbox2))


def face_im_gen(faces):
    for face in faces:
        yield face[0]


class TestAl(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), 'test_conf.yml')
        with open(path, 'r') as ymlfile:
            global cfg
            cfg = yaml.load(ymlfile)
        align = cfg['recognition']['face alignment']

        self.d_conf = align['dlib conf']
        self.mtcnn_conf = align['mtcnn conf']
        self.opencv_conf = align['opencv conf']
        self.faces = load_faces(os.path.join(os.path.dirname(__file__), 'faces/faces.txt'))

    def single_image_test(self, image, bbox_given, bboxes_result, name):
        # print('name: {0}\ngiven: {1}\nto test: {2}'.format(name, str(bbox_given), str(bboxes_result)))
        # print('images contain only 1 face, if more detected - it\'s a fail')
        save_face(image, bboxes_result, name)

        self.assertEqual(len(bboxes_result), 1)

        common = boxes_common(bbox_given, bboxes_result[0])
        # print('common: ' + str(common))

        self.assertGreaterEqual(common, 0.7)

    def alignment_test(self, gen, name):
        for i in range(len(self.faces)):
            self.single_image_test(
                self.faces[i][0],
                self.faces[i][1],
                next(gen)[1],
                self.faces[i][2].replace('raw', 'faces_'+name)
            )

    def test_dlib(self):
        self.alignment_test(dlib_align(face_im_gen(self.faces), self.d_conf), 'dlib')

    def test_opencv(self):
        self.alignment_test(opencv_align(face_im_gen(self.faces), self.opencv_conf), 'opencv')

    def test_mtcnn(self):
        self.alignment_test(mtcnn_align(face_im_gen(self.faces), self.mtcnn_conf), 'mtcnn')
