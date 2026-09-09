# Employee Attrition Prediction using MLP

A deep learning-based **Employee Attrition Prediction System** developed using Python and Scikit-learn's `MLPClassifier`.

The system analyzes employee-related factors and predicts whether an employee is likely to leave the organization or stay. The project uses a **Multilayer Perceptron (MLP)** with two hidden layers for classification.

## Project Overview

Employee attrition is an important problem for organizations because unexpected employee turnover can affect productivity, operational costs, and workforce planning.

This project builds a neural-network-based classification system that uses employee information such as age, income, experience, job satisfaction, work-life balance, overtime, and other factors to predict employee attrition.

The MLP network consists of two hidden layers:

```text
Input Layer
    ↓
Hidden Layer 1 — 8 neurons
    ↓
Hidden Layer 2 — 4 neurons
    ↓
Output Layer — Attrition Prediction
```

## Features Used

The model uses the following employee attributes:

* Age
* Monthly Income
* Years at Company
* Total Working Years
* Distance From Home
* Job Satisfaction
* Work-Life Balance
* OverTime
* Number of Companies Worked
* Training Times Last Year

### Target Variable

**Attrition**

```text
1 → Employee may leave the company
0 → Employee is likely to stay
```

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Multilayer Perceptron (MLP)
* StandardScaler
* Train-Test Split
* Confusion Matrix

## Machine Learning Workflow

The project follows these steps:

```text
Load Dataset
     ↓
Analyze Dataset
     ↓
Check Missing Values
     ↓
Identify Numerical & Categorical Features
     ↓
Convert Categorical Data into Numerical Data
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Create MLP Neural Network
     ↓
Train Model
     ↓
Calculate Training Accuracy
     ↓
Calculate Testing Accuracy
     ↓
Generate Confusion Matrix
     ↓
Plot Loss Curve
     ↓
Check Model Fit
     ↓
Predict Attrition for New Employees
```

## Data Preprocessing

The categorical variables are converted into numerical form before training.

For example:

```text
OverTime:
Yes → 1
No  → 0

Attrition:
Yes → 1
No  → 0
```

The input features are standardized using `StandardScaler` so that features with different numerical ranges can be processed effectively by the neural network.

## Train-Test Split

The dataset is divided into:

```text
70% → Training Data
30% → Testing Data
```

A fixed `random_state=42` is used to make the split reproducible.

## MLP Model

The project uses Scikit-learn's `MLPClassifier`.

Configuration:

```text
Hidden Layers : 2
Layer 1       : 8 neurons
Layer 2       : 4 neurons
Activation    : ReLU
Solver        : Adam
Maximum Iterations : 1000
Random State  : 42
```

The network learns the relationship between employee characteristics and attrition during the training process.

## Model Evaluation

The trained model is evaluated using:

### Training Accuracy

Measures how accurately the model predicts employee attrition on the training dataset.

### Testing Accuracy

Measures how accurately the model predicts attrition on previously unseen testing data.

### Confusion Matrix

The confusion matrix shows the classification results in terms of:

```text
True Negative (TN)
False Positive (FP)
False Negative (FN)
True Positive (TP)
```

### Loss Curve

The training loss curve is plotted using the `loss_curve_` attribute of `MLPClassifier`.

The curve helps visualize how the model's training loss changes across iterations.

## Overfitting and Underfitting Analysis

Training and testing accuracy are compared to determine the model's generalization behavior.

### Overfitting

If the training accuracy is significantly higher than the testing accuracy, the model may be overfitting.

Example:

```text
Training Accuracy : 99%
Testing Accuracy  : 75%
```

### Underfitting

If both training and testing accuracy are relatively low, the model may be underfitting.

Example:

```text
Training Accuracy : 70%
Testing Accuracy  : 68%
```

A small difference between training and testing accuracy generally indicates better generalization.

## Prediction on New Employees

The project includes a `PredictAttrition()` function that accepts five new employee records and predicts whether each employee is likely to leave or stay.

Example output:

```text
Employee 1 : Employee likely to stay
Employee 2 : Employee May leave the company
Employee 3 : Employee likely to stay
Employee 4 : Employee May leave the company
Employee 5 : Employee likely to stay
```

The new records are scaled using the same `StandardScaler` fitted on the training data before making predictions.

## Project Structure

```text
Employee-Attrition-Prediction-MLP/
│
├── EmployeeAttrition.py
├── Employee_Attrition.csv
└── README.md
```

## Requirements

Install the required Python libraries using:

```bash
pip install pandas scikit-learn matplotlib
```

## How to Run

1. Clone the repository.

2. Navigate to the project directory.

3. Make sure `Employee_Attrition.csv` is present in the same directory as the Python program.

4. Run:

```bash
python EmployeeAttrition.py
```

The program displays:

* Dataset shape
* Column names
* First five records
* Missing-value information
* Numerical and categorical features
* Converted categorical data
* Scaled data
* Training accuracy
* Number of training iterations
* Testing accuracy
* Confusion matrix
* Loss curve
* Predictions for five new employees

## Key Learning Outcomes

* Loading and analyzing datasets using Pandas
* Handling categorical variables
* Feature scaling for neural networks
* Splitting data into training and testing sets
* Building an MLP neural network
* Training a classification model
* Evaluating training and testing performance
* Generating a confusion matrix
* Visualizing the training loss curve
* Making predictions on unseen employee records
* Identifying potential overfitting and underfitting

## Conclusion

This project demonstrates how a **Multilayer Perceptron neural network** can be applied to employee attrition prediction. The system performs data preprocessing, feature scaling, neural network training, model evaluation, visualization, and prediction on new employee records in a complete machine learning workflow.

