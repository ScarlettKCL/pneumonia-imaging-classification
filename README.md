# pneumonia-imaging-classification
## A classification model to group lung x-ray images depending on whether the patient has Pneumonia or not using CNN architecture.
Can be run in a notebook or another environment that supports python.
### About
The goal of this project was to develop a better understanding of the different classification algorithms used within supervised learning available, by implementing such algorithms and comparing the test accuracy when used on a dataste of chest x-ray images from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) to classify whether or not the patient has pneumonia.

### Model Statistics
```
GaussianNB:
Validation Accuracy: 83.91%
Test Accuracy: 71.96%
Classification Report on Test Data:
              precision    recall  f1-score   support

         0.0       0.62      0.63      0.63       234
         1.0       0.78      0.77      0.77       390

    accuracy                           0.72       624
   macro avg       0.70      0.70      0.70       624
weighted avg       0.72      0.72      0.72       624

DecisionTreeClassifier:
Validation Accuracy: 88.03%
Test Accuracy: 75.16%
Classification Report on Test Data:
              precision    recall  f1-score   support

         0.0       0.82      0.44      0.57       234
         1.0       0.74      0.94      0.83       390

    accuracy                           0.75       624
   macro avg       0.78      0.69      0.70       624
weighted avg       0.77      0.75      0.73       624

Sequential:
Test accuracy: 85.26%
```
### Evaluation
Sequential CNN model:
```
model = keras.Sequential([
    keras.layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.25),
])
```
By investigating the accuracy of a Decision Tree, Gaussian Naive Bayes, and Sequential CNN model, it was found that the most effective model would be the Sequential CNN, given that it gave the highest test accuracy out of all the models, of 85.26%. The model was adjusted to improve performance by changing the dropout value to 0.25 as well as adding a rescaling and max pooling layer. To improve the model's performance, the number and type of layers within the architecture could further be experimented with to find the optimal choice, and even more types of classification models could be used, such as a ResNet or Random Forest model.
