# Cyber Attack Classification and Clustering

This project focuses on classifying cyber attacks using multiple machine learning algorithms and evaluating their performance. Additionally, clustering is performed to uncover hidden patterns in the dataset.

## Overview

In this project, we analyze network traffic data to classify cyber attacks using three different machine learning algorithms: Decision Tree Classifier, K-Nearest Neighbors (KNN), and Artificial Neural Network (ANN). We also perform clustering using K-Means to categorize the data into distinct groups.

### Steps Involved:
1. **Preprocessing**: The dataset is preprocessed by handling missing values, encoding categorical variables, and scaling numerical features.
2. **Feature Selection**: The most relevant features for classification are selected using correlation analysis with the target variable (`Attack_type`).
3. **Classification**:
    - **Decision Tree Classifier**: The dataset is classified using a decision tree, and the model’s performance is evaluated using accuracy, precision, recall, and F1 score.
    - **K-Nearest Neighbors (KNN)**: The KNN algorithm is trained with the optimal value of `k` and evaluated based on similar performance metrics.
    - **Artificial Neural Network (ANN)**: A neural network model is trained with optimized hyperparameters for better performance.
4. **Clustering**: K-Means clustering is applied to the dataset to identify inherent clusters. The Elbow Method is used to determine the optimal number of clusters.
5. **Performance Comparison**: A comparison is made between the three classification algorithms using a bar plot that visualizes the accuracy, precision, recall, and F1 score for each model.

## Libraries and Tools

The following libraries need to be installed to run the project:

```bash
pip install pandas numpy matplotlib scikit-learn
