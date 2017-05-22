# ARP resources

### Pose Estimation
- [**OpenPose: A Real-Time Multi-Person Keypoint Detection And Multi-Threading C++ Library**](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
  - Paper [here](https://arxiv.org/abs/1611.08050).
  - It's based on [this code](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

### Face Detection
- [**MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks**](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). 
  - [Paper](https://arxiv.org/abs/1604.02878). 
  - Caffe [code](https://github.com/kpzhang93/MTCNN_face_detection_alignment). 
  - TensorFlow [code](https://github.com/davidsandberg/facenet/tree/master/src/align).

- [**OpenFace (UC)**](http://www.cl.cam.ac.uk/research/rainbow/projects/openface/).
  - [Code](https://github.com/TadasBaltrusaitis/OpenFace).
  - [Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki).

- [**LibFaceDetection** (binares only, commercial license)](https://github.com/ShiqiYu/libfacedetection).

- [[Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/ijcv_deformable_tracking_review.pdf)] A Comprehensive Performance Evaluation of Deformable Face
Tracking “In-the-Wild”.

### Face Recognition (won't be used, but maybe there's useful code or insights)
- [**FaceNet: A Unified Embedding for Face Recognition and Clustering**](https://github.com/davidsandberg/facenet). 
  - Papers [here](https://arxiv.org/abs/1503.03832), [here](http://ydwen.github.io/papers/WenECCV16.pdf) and [here](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf). 
  - Instructions for training on custom images [here](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images).

- [**OpenFace (it is also based on FaceNet): A general-purpose face recognition library with mobile applications**](http://cmusatyalab.github.io/openface/)
  - Paper [here](http://elijah.cs.cmu.edu/DOCS/CMU-CS-16-118.pdf).
  - [Code](https://github.com/cmusatyalab/openface/).

### Age and Gender Recognition
- [**Age and Gender Classification using Convolutional Neural Networks**](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/).
  - [Paper](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf).
  - TensorFlow [code](https://github.com/dpressel/rude-carnie). 
  - Author's [help files](https://github.com/GilLevi/AgeGenderDeepLearning). 

- [**Deep expectation of real and apparent age from a single image without facial landmarks**](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). 
  - [Paper](https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf).
  - Datasets (faces only): [IMDB (7GB)](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) and [Wiki (1GB)](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar).

### Emotion Recognition
- Reuse thesis work based on RNN?
- [Recurrent Neural Networks for Emotion Recognition in Video](https://github.com/saebrahimi/Emotion-Recognition-RNN).
  - [Paper](http://www-etud.iro.umontreal.ca/~michals/pdf/emotion_rnns.pdf).
- [Deep learning API with emotion recognition application](https://github.com/mihaelacr/pydeeplearn).
  - Project [report](http://www.doc.ic.ac.uk/teaching/distinguished-projects/2014/mrosca.pdf).
  - [How to](https://github.com/mihaelacr/pydeeplearn/blob/master/code/webcam-emotion-recognition/Readme.md) run the code.

### Visual Trackers
- [**SANet: Structure-Aware Network for Visual Tracking**](http://www.dabi.temple.edu/~hbling/code/SANet/SANet.html).
  - [Paper](http://www.dabi.temple.edu/~hbling/publication/SANet.pdf).
  - Author's Matlab [code](http://www.dabi.temple.edu/~hbling/code/SANet/sanet_code.zip).
  
- [**MDNet: Multi-Domain Convolutional Neural Network Tracker**](http://cvlab.postech.ac.kr/research/mdnet/).
  - [Paper](https://arxiv.org/pdf/1510.07945v2.pdf).
  - Author's Matlab [code](https://github.com/HyeonseobNam/MDNet).
  - TensorFlow [code](https://github.com/AlexQie/MDNet) (check quality).
  
- [**ECO: Efficient Convolution Operators for Tracking**](http://www.cvl.isy.liu.se/research/objrec/visualtracking/ecotrack/index.html).
  - [Paper](https://arxiv.org/pdf/1611.09224v1.pdf).
  - Author's Matlab [code](https://github.com/martin-danelljan/ECO). GPU version coming soon?

### Datasets
- [WIDER FACE: A Face Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).
- [FDDB: Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/).
- [First Affect-in-the-Wild Challenge](https://ibug.doc.ic.ac.uk/resources). [[Videos](https://www.dropbox.com/s/uv3oq7qtyb4qxzi/train.zip?dl=1)] [[Annotations](https://www.dropbox.com/s/3ydatoxj5tirc37/cvpr_mean_annotations_train.zip?dl=1)]
