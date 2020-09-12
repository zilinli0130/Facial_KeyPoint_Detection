# Facial_KeyPoint_Detection
The facial keypoint detection project aims at combining computer vision tecniques and deep learning architecture (Convolution Neural Network) to generated the keypoints
to enclose the features of eyes, mouth, nose on a human face image. These applications include faicial tracking, facial filters and emotion recongnition. The 
trained neural network is able to predict the locations of human face and generate keypoints onto the human face given any images.

## Installation
1. Clone the repository and download the Jupyter Notebook through **https://jupyter.org/**. It is highly recommended to open and run this project through Jupyter Notebook
```
git clone https://github.com/zilinli0130/Facial_KeyPoint_Detection

```

2. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

3. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
 


## Usage

1. Since not all the images are in the same shape, the `1. Load and Visualize Data.ipynb` file is to load and transform the dataset (YouTube Faces Dataset) including both the images and keypoints 
into standard dimensions. After the transform process, the dimension of each image inside the dataset should be  `torch.size([1, 224, 224])` and the dimension of each keypoints set
corresponded to each image should be  `torch.size([68, 2])`. This tells us that the image is a grey image since the depth is only 1, and it contains 224x224 elements. 
Moreover, there is a single face and 68 keypoints, with coordinates (x, y), for that face.

2. The detail of training process for neural network contains in `2. Define the Network Architecture.ipynb`. Before the traning process, It is recommeded to double check if the image size is `torch.size([1, 224, 224])`
and the size of keypoints set is `torch.size([68, 2])`. Then the dataset needs to be split into training and testing parts. The batch size is `batch_size = 10` for both training and testing 
dataset. The network architecture is shown below:

                  Net(
                  (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))

                  (conv2): Conv2d(32, 64, kernel_size=(4, 4), stride=(1, 1))

                  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))

                  (conv4): Conv2d(128, 256, kernel_size=(2, 2), stride=(1, 1))

                  (fc1): Linear(in_features=36864, out_features=1000, bias=True)

                  (fc2): Linear(in_features=1000, out_features=136, bias=True)

                  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

                  (drop): Dropout(p=0.4)
                )
  I choose the MSELoss criterion and use the Adam optimization method. Since predicting facial keypoints is a regression task, the CrossEntropyLoss criterion which works well for calssification problem is not longer applicable.
  The MSELoss method measures the mean square error between each element and its target and determine how closed the element is toward the target.

3. The  `3. Facial Keypoint Detection, Complete Pipeline.ipynb` is for testing and validation to see how the trained neural network performs. It is highly recommended to
do hyperparameter such as `batch_size`, `epoch`, `learning rate` tuning and try to adjust the number of neural network layers to see if the performance of model improves or not  







      
