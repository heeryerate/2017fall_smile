## 2017fall_smile

### Intro

In this project, we aim to accomplish real time facial emotion recognition and possibly move to facial emotion generation.

refer to: [real-time facial emotion recognition](https://www.youtube.com/watch?v=GMy0Zs8LX-o), [face generation](https://www.youtube.com/watch?v=gXMwSzuCjLk)

### Goal

1. Facial emotion recognition
    + for given face image, output its emotion label (happy, sad, angry, etc.)
2. Real time facial emotion recognition
    + use web-camera to recognize facial emotion in real time
3. Facial emotion generation (would be awesome to have this done)
    + for given plain face, generate face with wanted emotion

### TimeLine

+ September 20th, 2017
    - Learn how to use `git` and work with `github`

         please try to modify the following two lines:
        * zhenyu: (*add something here*)
        * judy: (*add something here*) 
    - Use python package `opencv` to detect faces in a photo, something like
            <div align="center"><img width="50%" height="50%" src="http://www.compciv.org/files/images/homework/face-boxer-ellen.jpg"/></div>

        for given image including faces, they should be exported as single face images

        the link might be useful: [facial detection in 20 lines](https://realpython.com/blog/python/face-recognition-with-python/), feel free to use other tools
    - Find and download emotional faces datasets, something like
            <div align="center"><img width="30%" height="30%" src="http://www.basicknowledge101.com/photos/2015/emotions-3.jpg"/></div>

      Note the dataset you found should have uniform size, style and also labels regarding different emotions, the larger the dataset the better

      (please show some samples before downloading)
+ September 27th, 2017
    - Understand convolutional neural network (CNN)
    - Use `pytorch` to train a CNN for hand-written digits classification

        the link might be useful: [pytorch mnist example](https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb), feel free to use other sources
    - *We will have joint meeting with Martin this week, we are suppose to show something that works.*
+ October 4th, 2017
    - Start dealing with emotional faces dataset
         process dataset into appropriate form for training purpose
    - Learn how to use `polyps` and run our classification model on server, probably use `GPU`
+ October 11th, 2017
    - Have a workable facial emotions recognition (FER) model, means we should be able to recognize the emotion of given face image
    - Try deeper networks and larger dataset
    - *We will have joint meeting with Martin this week*
+ October 18th, 2017
    - Extend the work to web-camera, we should be able to export images from real time web-camera and recognize the face emotions in the image

      A real time web-camera implementation by OpenCV [Blog:](https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/)
    - Do real time FER
+ October 25th, 2017
    - Produce a visual demo for our real time facial emotions recognition, something like [real-time facial emotion recognition](https://www.youtube.com/watch?v=GMy0Zs8LX-o)
    - *We will have joint meeting with Martin this week*
+ November 1st, 2017
    - Try other networks architecture, like `resNet` and `denseNet`
    - Improve our FER model accuracy level
+ November 8th, 2017
    - Start work on facial emotion generation (FEG) model using `GAN` networks
    - Have a workable `GAN` demo from some existing projects
    - *We will have joint meeting with Martin this week*
+ November 15th, 2017
    - Apply `GAN` networks to our FEG model
+ November 22th, 2017
    - Produce a visual demo of FEG
    - *We will have joint meeting with Martin this week*
+ November 29th, 2017
    - :smile:

### Dependency

+ python basics/opencv/pytorch
+ basic linear algebra
+ neural networks

### What you will get from this project

+ Experience to work with `git`/`github`
+ A real touch to `deep learning`/`artificial intelligence`
+ A nice adding to you `CV` if your aim to a tech company
