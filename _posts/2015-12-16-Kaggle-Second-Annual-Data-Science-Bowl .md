---
title: "Kaggle Second Annual Data Science Bowl "
excerpt: "MRI Medical Image to Predict Cardiac using Deep Learning Approach"
header:
  teaser: "front_page.png"
categories: 
  - Deep Learning
  - Machine Learning
  - Python
tags: 
  - Deep learning
  - Pyhton
  - NumPy
  - SciPy
  - PYDICOM
---
### Project Description
This project involved to predict the heart failure using Medical MRI Images. This is very challenge and using deep learning to analysis 1140 patient dataset to calculate the ejection function (EF percentage). The datasets contain SAX-AXIS, four chamber view, three chamber view and two chamber view of heart. The cardiac function is measure by end-systolic and end-diastolic which means the size of the chamber of heart at the beginning and middle of each heartbeat. The final result should contain all patient 1140 dataset with separate volume of chamber in ml probability is calculated for all systolic and diastolic for all patients.  

### Responsibilities
    1. Responsibilities first read the .DICOM image python using PYDICOM package. The SAX .DICOM images is used for analysis
    2. First the image get auto segment for load all images from train, validate and validate, test.
    3. Using the label mapping using train.csv for train, validate and validate.csv for validate, test datasets.
    4. The images are get sliced and for each image the volume of the chamber is calculated by ROI and the Fourier series is applied all the slices.
    5. The volume of the chamber is calculated separately for systolic and diastolic. Then the solution submission.csv will contain all the patient systolic and diastolic volume probability value for 0ml to 599ml for all patient systolic and diastolic. 
    6. Organizing and coordinating the project development.

### Achievement
Achieved KAGGLER position in Kaggle and successfully submitted my solution in Second Annul Data Science Bowl 2015 competition.  

### Link
Check out my profile [Kaggle][BalajiN-Kaggle].

### Environment
Python, Deep Learning, Neural Network, Medical Images, MRI Images, OpenCV, DICOM, NUMPY, SCIPY, Fourier series, SOFTMAX function, ejection function, chamber views, Anaconda python.

### GitHub Gist Embed

An example of a Gist embed below.

{% gist  %}

Check out the for more info on how to get the most out of Balaji N. File all bugs/feature requests at [BalajiN GitHub repo][Balajin-gh]. If you have questions, you can ask me [Balaji Talk][BalajiN-talk].

[BalajiN-Kaggle]: https://www.kaggle.com/balajibi 
[Balajin-gh]:   https://github.com/balajincse
[BalajiN-talk]: mailto:balajincse@outlook.com
