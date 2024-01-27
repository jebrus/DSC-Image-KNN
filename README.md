# DSC-Image-KNN: K-Nearest Neighbors Image Classifier

## Overview
This class project focuses on processing images and implementing a k-nearest neighbors (KNN) classifier to categorize images. It includes functionalities for both image manipulation and classification using the KNN algorithm.

## Installation
Ensure you have Python installed on your system. Then, install the required packages using the following command:

`pip install numpy opencv-python`

Note: `cv2` is part of the OpenCV library, which is installed via `opencv-python`.

## Getting Started

### Preparing the Dataset
1. **Creating Image Objects:** Use the `img_read` method in `midqtr_project_runner.py` to create `RGBImage` objects.
2. **Dataset Creation:** Create a training dataset consisting of `(image, label)` pairs, where `image` is an `RGBImage` object and `label` is a string representing the image's label.

### Implementing the Classifier
1. **Initialize Classifier:** Instantiate an `ImageKNNClassifier`. Set the number of neighbors (`k`) as desired.
2. **Training the Classifier:** Train the classifier with your dataset using the `.fit(data)` method, where `data` is your training dataset.
3. **Making Predictions:** Predict the label of new images using the `.predict(image)` method.

## Image Manipulation
This project includes various image manipulation methods for `RGBImage` objects. Detailed documentation for each method is available in their respective docstrings in `midqtr_project.py`.

## Contribution
Feel free to fork this repository to contribute to the project. We welcome enhancements, bug fixes, and suggestions.

## License
CC0 1.0 Universal (CC0 1.0) Public Domain Dedication

The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.

You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

See the full CC0 legal code here: https://creativecommons.org/publicdomain/zero/1.0/legalcode
