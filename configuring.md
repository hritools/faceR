# Configuration

Default configuration file for the project rests at
faceR/config.yml.
It currently has the following structure: 
```
camera:
  provider: opencv # type: {text-stream, opencv}
  framerate: 30
  # use if provider is text-stream
  width: 640
  height: 480
  video device: 0
  receiver:
    port: '5555'
    ip: '10.42.0.41'

recognition:
  face alignment:
    # {mtcnn, dlib, opencv}
    provider: opencv
    # entries specific to mtcnn
    mtcnn conf:
      model folder: models/mtcnn
      minsize: 15
      threshold: [0.6, 0.7, 0.9]
      factor: 0.709
    # entries specific to dlib
    dlib conf:
    # entries specific to opencv
    opencv conf:
      cascade path: models/haarcascade_frontalface_default.xml
      scale factor: 1.2
      min neighbors: 5
      min width: 15
      min height: 20

  embedding:
    provider: facenet
    use ncs: true
    # NCS device configurations
    ncs conf:
      graph file: models/20180402-114759_ncs/model-20180402-114759_ncs.graph
      device to use: 0
      image size: 160
    # facenet configurations, not much at the moment
    facenet conf:
      model: models/20180402-114759/20180402-114759.pb
      image size: 160

  classification:
    # {test-classifier only, sry, subject to change}
    provider: test-classifier
    test-classifier conf:
      model: models/classifier/test_classifier.pkl

  tracking:
    provider: dlib
    frames: 7
    dlib conf:
      multiprocessing: False
      processes: 4

display:
  provider: opencv
```
* **Camera**: instructs on how to get the video feed, for most cases opencv will work. 
Otherwise you will have to use **serve-stream.py** to start streaming.
* **Recognition**: '*-model' is specifying the path to the models, obviously.
    * **face alignment**: TODO
    * **embedding**: TODO
    * **classification**: TODO
    * **tracking**: TODO
* **Display**: is used to specify where to pass the final product after 
processing the frame. Mostly the same as camera, but has no frame rate parameter. 
