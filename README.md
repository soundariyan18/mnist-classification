# EXPERIMENT 03: CONVOLUTIONAL DEEP NEURAL NETWORK FOR DIGIT CLASSIFICATION

## AIM
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## PROBLEM STATEMENT AND DATASET:
Problem Statement:<br/>
The task at hand involves developing a Convolutional Neural Network (CNN) that can accurately classify handwritten digits ranging from 0 to 9. This CNN should be capable of processing scanned images of handwritten digits, even those not included in the standard dataset.

Dataset:<br/>
The MNIST dataset is widely recognized as a foundational resource in both machine learning and computer vision. It consists of grayscale images measuring 28x28 pixels, each depicting a handwritten digit from 0 to 9. The dataset includes 60,000 training images and 10,000 test images, meticulously labeled for model evaluation. Grayscale representations of these images range from 0 to 255, with 0 representing black and 255 representing white. MNIST serves as a benchmark for assessing various machine learning models, particularly for digit recognition tasks. By utilizing MNIST, we aim to develop and evaluate a specialized CNN for digit classification while also testing its ability to generalize to real-world handwritten images not present in the dataset.

## NEURAL NETWORK MODEL


## DESIGN STEPS
STEP 1:
Import tensorflow and preprocessing libraries.

STEP 2:
Download and load the dataset

STEP 3:
Scale the dataset between it's min and max values

STEP 4:
Using one hot encode, encode the categorical values

STEP 5:
Split the data into train and test

STEP 6:
Build the convolutional neural network model

STEP 7:
Train the model with the training data

STEP 8:
Plot the performance plot

STEP 9:
Evaluate the model with the testing data

STEP 10:
Fit the model and predict the single input


## PROGRAM:
### Name: soundariyan MN
### Register Number: 212222230146

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()
print("soundariyan MN 212222230146")
metrics[['accuracy','val_accuracy']].plot()
print("soundariyan MN 212222230146")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("soundariyan MN 212222230146")
print(confusion_matrix(y_test,x_test_predictions))

print("soundariyan MN 212222230146")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('images.jpeg')
type(img)

img = image.load_img('images.png')
plt.imshow(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print("soundariyan MN 212222230146")
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img1 = image.load_img('images.jpg')
plt.imshow(img1)
img_tensor1 = tf.convert_to_tensor(np.asarray(img1))
img_28_gray1 = tf.image.resize(img_tensor1,(28,28))
img_28_gray1 = tf.image.rgb_to_grayscale(img_28_gray1)
img_28_gray_inverted1 = 255.0-img_28_gray1
img_28_gray_inverted_scaled1 = img_28_gray_inverted1.numpy()/255.0

x_single_prediction1 = np.argmax(model.predict(img_28_gray_inverted_scaled1.reshape(1,28,28,1)),axis=1)
print("soundariyan MN 212222230146")
print(x_single_prediction1)
plt.imshow(img_28_gray_inverted_scaled1.reshape(28,28),cmap='gray')
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![OUT,2](https://github.com/user-attachments/assets/8e4d47bb-6fa5-4cf0-80fb-51e4a017c8fb)
![OUT,3](https://github.com/user-attachments/assets/70ac04a4-e94d-4072-bb94-2b424bf69f99)





### Classification Report
![OUT,5](https://github.com/user-attachments/assets/b4c8b8c1-e9f9-45c1-b10e-dcf837d6f99a)



### Confusion Matrix
![OUT,4](https://github.com/user-attachments/assets/4633aaf6-010f-4b60-b8ab-2b3a06bffc78)




### New Sample Data Prediction

![Screenshot 2024-09-13 114839](https://github.com/user-attachments/assets/04bc51ac-caf2-4598-b5c8-446878cdeade)




## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
