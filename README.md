# Is that a 🐈 or a 🐕?

## PyTorch Convolutional Neural Network for Binary Classification.

Images of 8,000 cats and dogs, taken from [this Kaggle dataset](https://www.kaggle.com/datasets/chetankv/dogs-cats-images), were used to train a convolutional neural network for the task of binary image classification. In the end this simple network achieved 80% accuracy on the test set of 2,000 images.     

<img src="https://user-images.githubusercontent.com/79708390/204142349-8b6580a6-b453-4927-9675-98933d0c513a.jpg" height=224 width=224></img>
<img src="https://user-images.githubusercontent.com/79708390/204142529-be744e6e-bef0-4b06-8f0a-eee538efaced.jpg" height=224 width=224></img>
<img src="https://user-images.githubusercontent.com/79708390/204142618-23a06960-886a-4512-bddc-e319c6342dcd.jpg" height=224 width=224></img>
<img src="https://user-images.githubusercontent.com/79708390/204142383-eaae8bde-ca41-4683-96d2-32172abb644d.jpg" height=224 width=224></img>
<img src="https://user-images.githubusercontent.com/79708390/204142491-61d26faf-8acd-4df2-b086-79d858bf80f5.jpg" height=224 width=224></img>
<img src="https://user-images.githubusercontent.com/79708390/204142812-0caeb4da-7300-44dc-ac41-a4b5b04c84d5.jpg" height=224 width=224></img>


## Architecture

Convolutional Neural Networks are useful because they allow us to exploit the spatial correlation between pixels in the image to reduce the dimension of the input space. The first step is to extract the key features from the image. We achieve this by performing convolution operations using a kernel, mask or filter in the convolutional layers. Once these features are extracted, they are passed to a second network which uses several fully connected layers for classification.

![cnn](https://user-images.githubusercontent.com/79708390/204142288-ce99f74e-c225-4b25-b186-99c4793fa4b6.png)    

_CNN Architecture Diagram generated using:_ http://alexlenail.me/NN-SVG/AlexNet.html    
    
* To ensure consistent model input, these RGB images are first transformed in preprocessing to dimensions 3×256×256. 
* The feature extraction consists of 5 convolutional layers and 4 pooling layers all using ReLU() activation. 
* Batch normalisation was applied after every convolution operation to assist the training of the model. 
* The output of the final pooling layer is flattened and fed into the classifier, comprised of 4 fully connected layers using tanh() activation. 



## Brief Notes on the Motivation for Convolutional Neural Networks

Imagine a supervised learning problem where you have one input feature per pixel in an image. For a megapixel image your input space would be on the order of millions. This is computationally intractable for a standard neural network to learn. CNNs are handy because they reduce this high dimensional input by performing convolutions. 
Using a sliding kernel, they reduce the input space.    

![image](https://user-images.githubusercontent.com/79708390/204163695-3ee8b607-ec98-47d4-8b01-971e13c1bdcf.png)    
(Img Source: https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)

Pixels in an image are highly correlated according to their proximity to one another. By sliding this convolutional filter across the image, we hope that the resulting convolved outputs can successfully capture some of that spatial dependence. We can use a pooling layer which helps us to more clearly ascertain the features which correspond to these spatial structures found in the image. This process of convolving the input image to identify key spatials tructures is called feature extraction.
In other words by performing this convolutional operation, we are creating a feature map. We pass these extracted features along to a classification network.    

This also allows us to be more resilient to slight modifications or perturbations in the image. If we had a simple one pixel translation, a standard neural network would be highly sensitive to these small changes, because at a purely numerical level all of the inputs have suddenly changed. By contrast, using this convolutional approach, since we are capturing the broader spatial structures within regions of the image the neural network is more robust to these small perturbations.    

Each filter will have a specific shape or pattern that it is trying to detect. A key step is the pass the convolution outputs to a pooling layer. This is necessary to further reduce the dimensionality of the data. For example a 2x2 MaxPooling filter will consider a 2x2 subset of an input, take the maximum value of that 2x2 subset, and associate that to a single 1x1 subset of the output. The particular number of convolutional filters and the scalar values in those filters is optimised during the backpropagation.     

![image](https://user-images.githubusercontent.com/79708390/204163584-5761c953-19be-4d05-b5b1-1ee5efbe5d95.png)
(Img Source: https://towardsdatascience.com/convolution-neural-networks-a-beginners-guide-implementing-a-mnist-hand-written-digit-8aa60330d022)
