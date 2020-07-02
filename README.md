# Indian-Scene-Text-Detection

The Indian scene text detection model is developed as part of the work towards [Indian Signboard Translation Project](https://ai4bharat.org/articles/sign-board) by [AI4Bharat](https://ai4bharat.org/). I worked on this project under the mentorship of [Mitesh Khapra](http://www.cse.iitm.ac.in/~miteshk/) and [Pratyush Kumar](http://www.cse.iitm.ac.in/~pratyush/) from IIT Madras.

Indian Signboard Translation  involves 4 modular tasks:
1. **`T1`: Detection:** Detecting bounding boxes containing text in the images
2. *`T2`: Classification:* Classifying the language of the text in the bounding box identifed by `T1`
3. *`T3`: Recognition:* Getting the text from the detected crop by `T1` using the `T2` classified recognition model
4. *`T4`: Translation:* Translating text from `T3` from one Indian language to other Indian language

![Pipeline for sign board translation](../master/Images/Pipeline.jpg)
> Note: `T2`: Classification is not updated in the above picture


# Dataset

[Indian Scene Text Detecttion Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset#d1-detection-dataset) is used for training the detection model and evaluation


# Model


# Training


# Performance


# Code


### Related Links:
1. [Indian Signboard Translation Project](https://ai4bharat.org/articles/sign-board)
2. [Indian Scene Text Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset)
3. [Indian Scene Text Detection](https://github.com/GokulKarthik/Indian-Scene-Text-Detection)
4. [Indian Scene Text Classification](https://github.com/GokulKarthik/Indian-Scene-Text-Classification)
5. [Indian Scene Text Recognition](https://github.com/GokulKarthik/Indian-Scene-Text-Recognition)


### References:
