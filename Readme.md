# Emotion Intensity Prediction Project

## Overview

This project explores the task of predicting emotion intensity in text data using various regression models. The goal is to develop models capable of accurately predicting the intensity of emotions such as joy, sadness, anger, and fear based on textual input.

For more information about the problem statement and competition, please visit [here](https://competitions.codalab.org/competitions/16380) and refer to the provided [documentation](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html).

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Open your terminal and navigate to the project directory.
3. Run the command `python3 run.py` to execute the project.

## Data

The dataset consists of tweets labeled with real-valued scores indicating the degree of emotion intensity felt by the speaker. Separate datasets are provided for training, development, and testing purposes, with and without intensity labels.

You can access the transformed dataset in the `database` folder, along with the raw datasets. The `models` directory contains all regression models that have been performed. The `results` directory consists of all the predicted data from different models.

## Models Used

### Statistical Machine Learning Models:
1. Lasso Regression
2. Ridge Regression
3. SVM Regression
4. Decision Tree
5. Random Forest

### Deep Learning Models:
1. Classical Neural Networks (Multilayered Perceptron)
2. Convolutional Neural Network (CNN)

## Results

| Model                                     | Train Mean Squared Error | Train Mean Absolute Error | Test Mean Squared Error | Test Mean Absolute Error |
|-------------------------------------------|--------------------------|---------------------------|-------------------------|--------------------------|
| Lasso Regression                          | 0.0365                   | 0.1561                    | 0.0396                  | 0.1642                   |
| Ridge Regression                          | 0.0060                   | 0.0608                    | 0.0326                  | 0.1436                   |
| SVM Regression                            | 0.0068                   | 0.0759                    | 0.0285                  | 0.1372                   |
| Decision Tree                             | 0.0252                   | 0.1260                    | 0.0352                  | 0.1501                   |
| Random Forest                             | 0.0046                   | 0.0526                    | 0.0316                  | 0.1398                   |
| Classical Neural Networks (MLP)           | 0.0021                   | 0.0336                    | 0.0364                  | 0.1518                   |
| Convolutional Neural Network (CNN)        | 0.0367                   | 0.1556                    | 0.0404                  | 0.1652                   |

## Conclusion

This project demonstrates the effectiveness of various regression models in predicting emotion intensity in text data. While traditional machine learning models perform competitively, deep learning models show promise for further exploration. Moving forward, continued refinement and experimentation are essential to enhance the accuracy and applicability of emotion intensity prediction models.
