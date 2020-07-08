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

[Indian Scene Text Detection Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset#d1-detection-dataset) is used for training the detection model and evaluation. Axis-Aligned Bounding Box representation of the text boxes are used. 


# Labels
The score map for an image is the region with in the shrinked bounding box. The geometry map at a point inside the bounding box represents the distance of that point to the left, top, right and bottom boundaries respectively.

![Sample-X-Y](../master/Images/Sample-X-Y.png)


# Model
The fully convolutional neural network proposed in the paper titled "An Efficient and Accurate Scene Text Detector" ([EAST](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf)) is used to predict the word instance regions and their geometries. The following two variants of the model are experimented:

1. **`M1`:** Pretrained [VGG-16 net](https://arxiv.org/pdf/1409.1556.pdf) as a feature extractor.
  It produces output in the reduced dimensions by a factor of 4.
  * Input Image Shape: [320, 320, 3]
  * Output Score Map Shape: [80, 80, 1]
  * Output Geometry Map Shape: [80, 80, 4]

2. **`M2`:** [U-Net](https://arxiv.org/pdf/1505.04597.pdf) for feature extractor and merging.
  It produces per pixel predictions of text regions and geometries.
  * Input Image Shape: [320, 320, 3]
  * Output Score Map Shape: [320, 320, 1]
  * Output Geometry Map Shape: [320, 320, 4]  
  
Non-Maximal Supression is performed to remove the overlapping bounding boxes with the maximum permitted IoU threshold of 0.1.

For detailed model architecture, check the file [model.py](../master/model.py)

**Sample Input-Output**

![Sample-X-Y-Pred](../master/Images/Sample-X-Y-Pred.png)


# Training
`M1` & `M2` converged to simliar score and geometry losses after training for a specific number of epochs. As `M1`is significantly efficient in memory and computation, it is selected over `M2`. The detection model is trained for 30 epochs. The model weights are saved every 3 epochs and you can find them in the `Models`[../master/Models] directory.

The final hyperparameters can be accessed in [config.yaml](../master/config.yaml)

![Training Loss](../master/Images/Training-Loss.png)


# Performance
The lowest validation loss is observed in epoch 21. Hence, the model [`Models/EAST-Detector-e21.pth`](../master/Models/EAST-Detector-e24.pth) is used to evaluate the detection performance. 

**Sample Detections:**

![Sample Detections](../master/Images/Sample-Detections.png) 

# Code
* Model: [model.py](../master/model.py)
* Training: [1-Indian-Scene-Text-Detection-Training](../master/1-Indian-Scene-Text-Detection-Training.ipynb)
* Training Visualisation: [2-MLFlow-Training-Visualisation](../master/2-MLFlow-Training-Visualisation.ipynb)
* Prediction: [3-Indian-Scene-Text-Detection-Prediction](../master/3-Indian-Scene-Text-Detection-Prediction.ipynb)
* Evaluation: [4-Indian-Scene-Text-Detection-Evaluation](../master/4-Indian-Scene-Text-Detection-Evaluation.ipynb)


### Related Links:
1. [Indian Signboard Translation Project](https://ai4bharat.org/articles/sign-board)
2. [Indian Scene Text Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset)
3. [Indian Scene Text Detection](https://github.com/GokulKarthik/Indian-Scene-Text-Detection)
4. [Indian Scene Text Classification](https://github.com/GokulKarthik/Indian-Scene-Text-Classification)
5. [Indian Scene Text Recognition](https://github.com/GokulKarthik/Indian-Scene-Text-Recognition)


### References:
1. https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf
2. https://arxiv.org/pdf/1505.04597.pdf
3. https://github.com/liushuchun/EAST.pytorch
4. https://github.com/GokulKarthik/EAST.pytorch
5. https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
