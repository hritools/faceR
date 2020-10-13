import os
import pickle
import imageio
from collections import namedtuple

from faceR.embedding import embed
from faceR.conf import setup

DataSetPic = namedtuple("DataSetPic", "path person embedding")
Person = namedtuple("Person", "name directory pics")


class DataSet:
    people = []

    def __init__(self, directory, image_size):
        self.image_size = image_size
        dirs = os.listdir(directory)
        dirs.sort()
        for cls in dirs:
            pics = []
            for file in os.listdir(os.path.join(directory, cls)):
                pics.append(DataSetPic(
                    path=os.path.join(directory, cls, file),
                    person=cls,
                    embedding=None
                ))
            if pics:
                self.people.append(Person(name=cls, directory=os.path.join(directory, cls), pics=pics))

    def embed(self):
        pic_gen = []
        emb_gen = embed(pic_gen)

        for person in self.people:
            for i in range(len(person.pics)):
                img = imageio.imread(person.pics[i].path)

                if img.shape == (self.image_size, self.image_size, 3):
                    pic_gen.append((img, [[0, 0, self.image_size, self.image_size]]))
                    embedding = next(emb_gen)[0]
                    print('Calculating embeddings for %s' % person.pics[i].path)
                else:
                    embedding = None
                    print('Error: not expected pic dimensions! skipping embeddings for %s' % person.pics[i].path)
                person.pics[i] = DataSetPic(
                    path=person.pics[i].path,
                    person=person.pics[i].person,
                    embedding=embedding
                )

    def save_as_file(self, file):
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_as_dataset(self):
        for person in self.people:
            if not os.path.exists(person.directory):
                os.makedirs(person.directory)
                for image in person.pics:
                    imageio.imwrite(os.path.join(person.directory, os.path.basename(image.path)), image)

    @classmethod
    def load(cls, file):
        in_file = open(file, 'rb')
        return pickle.load(in_file)

    def get_all_images(self):
        return [pic for person in self.people for pic in person.pics]


if __name__ == '__main__':
    setup("config.yaml", 'logging.yaml')
    ds = DataSet('faces')
    ds.save_as_file('faceR/data/aligned_ov.pkl')
    print(ds)
