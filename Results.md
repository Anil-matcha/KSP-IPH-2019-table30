# Problem Statement

Find the photo of a missing person from police database or string of other official databases or some social media platforms and internet in general. Your task is to develop an App to capture a photo and search for the same in official databases using an optimized facial recognition algorithm.

# Data:

Set of 6000 Images of Missing/Arrested/Unnatural Death/ Wanted people

# Approach:

**Face Detection**

We divided the problem into 2 stages
1. 1st stage consists of Face detection i.e detecting a face and it's coordinates in an image. We did this by using MTCNN architecture.

![](https://miro.medium.com/max/2506/1*ICM3jnRB1unY6G5ZRGorfg.png)

MTCNN consists of 3 networks P-Network, R-Network and O-Network. P-Network provides basic bounding boxes by doing a 12x12 sliding window which then are fine-tuned by R-Network which gives tuned bounding boxes and confidence scores for each box. O-Network takes the outputs of R-Network and gives an output of bounding boxes, confidence scores and facial landmarks. 
The landmarks obtained from the network are helpful in correcting posture of face which is needed for face recognition

![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/03/Pipeline-for-the-Multi-Task-Cascaded-Convolutional-Neural-Network-862x1024.png)

**Face Recognition**
For face recognition we used a Inception-Resnet based architecture to recognize the faces sent by detector. Detector crops the image across the bounding box which is sent to recognizer.

![](https://1.bp.blogspot.com/-O7AznVGY9js/V8cV_wKKsMI/AAAAAAAABKQ/maO7n2w3dT4Pkcmk7wgGqiSX5FUW2sfZgCLcB/s1600/image00.png)

The network is trained to give a 512 vector for every image input which is unique for every class. This is achieved by training the network using a triplet loss which makes the vector output of each class farther from each other. 

**The advantage of such an architecture is new classes can be added on the fly, there is no need for network change**

![](https://miro.medium.com/max/2480/1*lR_73E22fvaOfR5pzfOkdQ.png)

Once the network is trained to identify the classes and make them distant, when a new image is provided it is matched with existing classes by comparing its vector output to every class vector output. The distance metric is subjective and we tried various things like euclidean distance, SVM, cosine distance, LSH etc and we found euclidean to provide the best results.

# Blockers when we started:

*  The Dataset Provided is of low resolution, hazzy, and inadequate in Number. 
*  The above blocker blocks us down to avoid OpenCv approach and directs us to Deep Neural Networks(DNN). Training a DNN requires a Huge amount of data. 

# Data Augmentation Techniques used:

We did a grid search to find the augmentation techniques:
* Random Crop.
* Random FLip_LR
* Guassian Noise
* Illumenance
* color channel changes

# Results:

We have developed a **robust** model which is tackle the following:

## Similar Face Matching:

The main task of the competition is to match images of people dead by natural causes with images from arrested/wanted/missing people images and to provide best match. As we can see the outputs are pretty awesome considering the limited amount of data and the noise present in most of the dead images

### Output
![Similar Match - 844](Results_Images/Similarity_Matching/844.png)
![Similar Match - 844](Results_Images/Similarity_Matching/844_t.png)

![Similar Match - 848](Results_Images/Similarity_Matching/848.png)
![Similar Match - 848](Results_Images/Similarity_Matching/848_t.png)

![Similar Match - 1076](Results_Images/Similarity_Matching/1076.png)
![Similar Match - 1076](Results_Images/Similarity_Matching/1076_t.png)

## Facial Recognition over given database:

We have developed an Algorithm which gives **97.9 % accuracy**  which is comparable to the state of the art given it was trained over a very clean and good dataset. 

### Output - Person 1
![person1](Results_Images/Face_recognition_given_dataset/CR\ NO\ 132\ #_a1ee9974-34ae-42f8-a914-05cb2f4c08971251.jpg)
![person1 - match]

### Output - Person 2
![person2]
![person2 - match]

## Facial Recognition over internet data

We wanted to check the model performance on unseen type of data. So we have scraped web for data of 30 celebrities and were able to produce similar results as above mentioned

### Output


## Facial Recognition on different hardware data:

We wanted to test model reliability by checking with images taken from different hardware devices such as multiple phone cameras, webcam from computer etc. and our model was able to recognize all the variants with reliable accuracy.

# Train

**Train 1**
![chirag train](Results_Images/Face_recognition_different_hardware/train/chirag/20191117_095752.jpg)

**Train 2**
![shubham train](Results_Images/Face_recognition_different_hardware/train/shubham/20191117_090952.jpg)

# Test

**test 1**

![chirag test](Results_Images/Face_recognition_different_hardware/test/chirag/2019-11-17-102234.jpg)
![chirag test 2](Results_Images/Face_recognition_different_hardware/test/chirag/2.jpg)

**test 2**

![shubham_test](KSP-IPH-2019-table30/Results_Images/Face_recognition_different_hardware/test/shubham/2019-11-17-091132.jpg)


Results - 100% over different hardware specifications.

## Real Time Facial Recognition:

We ran the model over a live webcam and the model was able to work at a speed of 5 - 10 Frames Per Second(FPS).

# Choice of Technology:
* Tensorflow, Keras
* Google colab
* Open source India face dataset
* Facenet, MTCNN
