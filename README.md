# HISS repo

## Face recognition
This part explains face recognition side of the project, and is basing on 
[FaceNet](https://github.com/davidsandberg/facenet), OpenCV and MTCNN.

### Requirements
* PyYAML
* pyzmq
* numpy
* tensorflow
* scipy
* scikit-learn
* opencv-python
* Pillow
* dlib

### Distribution
You can use the project as a separate python package, it rests at our 
[PyPi](https://unihost-dg03.uni.innopolis.ru/nexus/repository/Python-repo/simple/facer/).

Before uploading your distribution change 'name', 'author', 'version' parameters 
in setup.py script.
To build a package yourself run 
`python setup.py sdist bdist_wheel` 
in root directory of the project.

Python package `twine` can automate upload process, execute `pip install twine` 
to install it.
To upload a package you have built you need to download certificate, you can use 
your web-browser.

The following command will upload the package for you.
`python -m twine upload --cert downloaded.crt --repository-url https://unihost-dg03.uni.innopolis.ru/nexus/repository/Python-repo/ dist/*`
### Configuration
The default configuration allows you to run the code on a PC, with 
reasonably low framerate. In case you want to tune anything, 
detailed explanation of configuration is provided in a 
[separate file](configuring.md).



### Models
##### FaceNet
As per FaceNet models, you can use pre-trained models from 
[davidsandberg/facenet](https://github.com/davidsandberg/facenet#pre-trained-models).
The other models will be uploaded somewhere soon.

### Data processing
If you want to get a better quality dataset to train your classifier you can have 
a look at a more [detailed guide]() on how to process raw images can be found.

##### Organize faces
 The dataset of faces you want to recognise should have the following structure:
```
face_DB/raw
├── ID1
│     ├── ID1_001.jpg
│     ├── ID1_002.jpg
│     ├── ID1_003.jpg
│     ├── ID1_004.jpg
│     └── ID1_005.jpg
├── ID2
│     ├── ID2_001.jpg
│     ├── ID2_002.jpg
│     ├── ID2_003.jpg
│     ├── ID2_004.jpg
│     └── ID2_005.jpg
├── ID3
│     ├── ID3_001.jpg
...
...
```

##### Train the classifier 
Align faces:
```
python run.py align <raw_images_dir> <save_dir>
```
* <raw_images_dir>: where to get raw images from;
* <save_dir>: where to store aligned faces.

Train a classifier:
```
python run.py train_classifier <aligned_dir> <classifier_name>
```
* <aligned_dir>: where to get aligned faces;
* <classifier_name>: where the trained classifier is saved.

### How do I run?
To run everything locally:
```
python test_run.py
```


## Inspiration
This project is heavily influenced by FaceNet, 
and particularly, tensorflow implementation of it by 
[davidsandberg](https://github.com/davidsandberg/facenet).
The following scripts and files were taken from this repository:
* facenet.py
* detect_face.py
* align_dataset_mtcnn.py
* classifier.py
* models/mtcnn/
