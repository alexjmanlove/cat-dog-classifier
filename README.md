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

![cnn](https://user-images.githubusercontent.com/79708390/204142288-ce99f74e-c225-4b25-b186-99c4793fa4b6.png)    

CNN Architecture Diagram generated using: http://alexlenail.me/NN-SVG/AlexNet.html    
    
To ensure consistent model input, these RGB images were transformed in preprocessing to dimensions 3×256×256. The feature extraction consists of 5 convolutional layers and 4 pooling layers all using ReLU() activation. Batch normalisation was applied after every convolution operation to assist the training of the model. The output of the final pooling layer is flattened and fed into the classifier, comprised of 4 fully connected layers using tanh() activation. Implementation details are available in the .ipynb file.
