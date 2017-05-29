# ARP resources

### Pose Estimation
- [**OpenPose: A Real-Time Multi-Person Keypoint Detection And Multi-Threading C++ Library**](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
  - Paper [here](https://arxiv.org/abs/1611.08050).
  - It's based on [this code](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

- [[Paper](https://arxiv.org/abs/1608.03932)] *Human Pose Estimation from Depth Images via Inference Embedded Multi-task Learnings*.


### Face/Object Detection
- [**Mask R-CNN** (non-official TensorFlow code)](https://github.com/CharlesShang/FastMaskRCNN).
  - [Paper](https://arxiv.org/abs/1703.06870).
  - PyTorch [code](https://github.com/felixgwu/mask_rcnn_pytorch).

- [**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** (python Caffe)](https://github.com/rbgirshick/py-faster-rcnn).
  - [Paper](https://arxiv.org/abs/1506.01497).
  - Additional TensorFlow implementations [here](https://github.com/CharlesShang/TFFRCNN), [here](https://github.com/endernewton/tf-faster-rcnn) and [here](https://github.com/smallcorgi/Faster-RCNN_TF).
  - PyTorch implementation [here](https://github.com/longcw/faster_rcnn_pytorch).
  - Recent method improvement: 
    - *Face Detection using Deep Learning: An Improved Faster RCNN Approach* [[Paper](https://arxiv.org/abs/1701.08289)] [[Non-official code (python Caffe](https://github.com/playerkk/face-py-faster-rcnn)].

- [**MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks**](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). 
  - [Paper](https://arxiv.org/abs/1604.02878). 
  - Caffe [code](https://github.com/kpzhang93/MTCNN_face_detection_alignment). 
  - TensorFlow [code](https://github.com/davidsandberg/facenet/tree/master/src/align).
  
- [**OpenFace (UC)**](http://www.cl.cam.ac.uk/research/rainbow/projects/openface/).
  - [Code](https://github.com/TadasBaltrusaitis/OpenFace).
  - [Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki).

- [[Paper](https://arxiv.org/abs/1705.02402v1)] *Face Detection, Bounding Box Aggregation and Pose Estimation for Robust Facial Landmark Localisation in the Wild*
- [[Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/ijcv_deformable_tracking_review.pdf)] *A Comprehensive Performance Evaluation of Deformable Face Tracking “In-the-Wild”*.

### Face Recognition
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
  
- [[Paper](https://arxiv.org/abs/1705.04515v1)] *Spatial-Temporal Recurrent Neural Network for Emotion Recognition* (12 May 2017).

## Recommender Systems
- Stanford Infolab [notes](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf).
- [List](https://github.com/grahamjenson/list_of_recommender_systems) of recommender systems.
- Deep learning [resources](https://github.com/robi56/Deep-Learning-for-Recommendation-Systems) for recommender systems.

- [[Paper](https://arxiv.org/abs/1606.07792v1)] *Wide & Deep Learning for Recommender Systems* (Google).


### Visual Trackers
- [**SANet: Structure-Aware Network for Visual Tracking**](http://www.dabi.temple.edu/~hbling/code/SANet/SANet.html).
  - [Paper](http://www.dabi.temple.edu/~hbling/publication/SANet.pdf).
  - Author's Matlab [code](http://www.dabi.temple.edu/~hbling/code/SANet/sanet_code.zip).
  
- [**MDNet: Multi-Domain Convolutional Neural Network Tracker**](http://cvlab.postech.ac.kr/research/mdnet/).
  - [Paper](https://arxiv.org/pdf/1510.07945v2.pdf).
  - Author's Matlab [code](https://github.com/HyeonseobNam/MDNet).
  - TensorFlow [code](https://github.com/AlexQie/MDNet) (non-official implementation).
  
- [**ECO: Efficient Convolution Operators for Tracking**](http://www.cvl.isy.liu.se/research/objrec/visualtracking/ecotrack/index.html).
  - [Paper](https://arxiv.org/pdf/1611.09224v1.pdf).
  - Author's Matlab [code](https://github.com/martin-danelljan/ECO). GPU version coming soon?

### Action Recognition
- [[Paper](https://arxiv.org/abs/1704.07333)] *Detecting and Recognizing Human-Object Interactions*.

- [**Contextual Action Recognition with R\*CNN**](https://github.com/gkioxari/RstarCNN).
  - [Paper](https://arxiv.org/abs/1505.01197).

- Georgia Gkioxari [page](https://people.eecs.berkeley.edu/~gkioxari/).

### Datasets
- [WIDER FACE: A Face Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).
- [FDDB: Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/).
- [LFW: Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).
- [First Affect-in-the-Wild Challenge](https://ibug.doc.ic.ac.uk/resources). [[Videos](https://www.dropbox.com/s/uv3oq7qtyb4qxzi/train.zip?dl=1)] [[Annotations](https://www.dropbox.com/s/3ydatoxj5tirc37/cvpr_mean_annotations_train.zip?dl=1)]
- **Facial Expression** datasets list [here](https://en.wikipedia.org/wiki/Facial_expression_databases).
