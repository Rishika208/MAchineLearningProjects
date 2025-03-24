# Breast Cancer Classification with Bagging and K-Nearest Neighbors

This code demonstrates the use of BaggingClassifier and KNeighborsClassifier from scikit-learn to classify breast cancer data.

## Data

The code uses the breast cancer dataset from scikit-learn's `load_breast_cancer` function. This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## Workflow

1. **Data Loading and Preprocessing:**
   - Loads the breast cancer dataset.
   - Creates a Pandas DataFrame for easier data manipulation.
   - Splits the data into input features (X) and target variable (y).

2. **Model Training and Evaluation:**
   - Splits the data into training and testing sets.
   - Creates a KNeighborsClassifier model and trains it on the training data.
   - Evaluates the KNN model's accuracy on the testing data.
   - Creates a BaggingClassifier model with KNeighborsClassifier as the base estimator.
   - Trains the BaggingClassifier model on the training data.
   - Evaluates the BaggingClassifier model's accuracy on the testing data.

3. **Prediction:**
   - Predicts the target variable using both the KNN and BaggingClassifier models on the testing data.

## Dependencies

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Usage

1. Make sure you have the necessary dependencies installed.
2. Run the code in a Jupyter Notebook or Python environment.
3. The code will print the accuracy scores of both models.
