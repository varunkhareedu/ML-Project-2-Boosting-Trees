# Project 2

**Illinois Institute of Technology CS584 - Machine Learning (Spring 2025)**

Team Members:

* A20552681 - Naga Sunith Appasani
* A20536596 - Varun Khareedu
* A20546720 - Venkata Gandhi Varma Thotakura
* A20555681 - Pardha Saradhi Bobburi

## Boosting Trees

Implement again from first principles the gradient-boosting tree classification algorithm (with the usual fit-predict interface as in Project 1) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1. In this assignment, you'll be responsible for developing your own test data to ensure that your implementation is satisfactory. (Hint: Use the same directory structure as in Project 1.)

The same "from first principals" rules apply; please don't use SKLearn or any other implementation. Please provide examples in your README that will allow the TAs to run your model code and whatever tests you include. As usual, extra credit may be given for an "above and beyond" effort.

As before, please clone this repo, work on your solution as a fork, and then open a pull request to submit your assignment. *A pull request is required to submit and your project will not be graded without a PR.*

Put your README below. Answer the following questions.

---

## Project File Structure

```
BoostingTrees/
├── generate_classification_data.py               # Original generator script
├── .gitignore
├── README.md                                     # Project documentation
├── BoostingTrees/
│   ├── generate_test_classification_data.py      # NEW: Script to generate test CSVs (collinear_class.csv and small_test_class.csv)
│   │
│   ├── model/
│   │   ├── GradientBoostingClassifier.py         # Core implementation
│   │
│   ├── tests/
│   │   ├── class_test_data.csv                   # General test dataset
│   │   ├── collinear_class.csv                   # Collinearity test dataset (binary)
│   │   ├── small_test_class.csv                  # Small 2-class dataset
│   │   ├── test_GradientBoostingClassifier.py    # PyTest suite
│   │   ├── __init__.py
```

---

## How to Set Up

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install numpy pandas pytest
```

---

## Generate Test Data

Run the provided script to generate synthetic binary classification data:

```bash
python generate_classification_data.py -N 100 -m 1.5 -2.0 -b 0.5 -rnge -3 3 -seed 42 -output_file tests/class_test_data.csv

```

---

## Run Tests

Run the test cases to validate the classifier:

```bash
pytest -s BoostingTrees/tests
```
For testing the different CSV files, you can change the file path of the csv to different csv in test_GradientBoostingClassifier.py 

Expected output:
```
Classification accuracy from CSV data: 0.84
.
```

---

## Example Usage

```python
from BoostingTrees.model.GradientBoostingClassifier import GradientBoostingClassifier
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5)
model.fit(X, y)
print(model.predict(X))
```

---

## Notes

- The model uses decision stumps (depth-1 trees) for gradient updates.
- It minimizes **log-loss** using a **forward stagewise additive model**.
- You can adjust:
  - `n_estimators`: number of boosting rounds
  - `learning_rate`: learning step size

---

* What does the model you have implemented do and when should it be used?

The employed algorithm is a Gradient Boosting Tree Classifier, implemented from scratch using NumPy. It is a type of ensemble learning method which takes a series of weak models (our case being shallow decision stumps) and combines them sequentially to give a strong classifier. At each iteration, the model is fit from the model's so-far errors (gradients) with the target function being log-loss. As time progresses, the ensemble improves by giving more importance to difficult-to-classify instances.

This model is to be employed when you are dealing with binary classification problems and require a model that is interpretable and also versatile. It is particularly valuable when the data involve non-linear interactions or relationships among the features that cannot be revealed by smaller models. Gradient boosting is robust with the default settings and can be tuned to prevent overfitting, thus why it's a favorite among practical machine learning uses in finance, medicine, marketing, and other fields.

* How did you test your model to determine if it is working reasonably correctly?

To make sure the Gradient Boosting Tree Classifier is functioning properly, We tested it with both synthetic data and a custom-generated CSV dataset. First, We employed a small, hardcoded dataset with an obvious binary decision boundary to check that the model was able to learn and split the classes correctly. Next, We generated a more realistic dataset using a custom Python script that produces binary labels from logistic probability — simulating real classification behavior.

We implemented unit tests with pytest to check output shape, prediction consistency, and a minimum level of expected accuracy. These tests ensure that the model is at least 80% accurate on the data it generates. We also printed out predicted labels and probabilities to see learning progress visually. We also cross-checked my model's results with anticipated outcomes based on theoretical gradient boosting behavior, specifically that predictions get better with increasing estimators and proper learning rates. This rigorous testing process ensured the correctness and reliability of the model.

* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

In this Gradient Boosting Tree Classifier, two of the most important parameters are made available to users for tuning: n_estimators and learning_rate.

n_estimators determines the number of boosting rounds (i.e., how many weak learners will be added). Raising this enables the model to learn more sophisticated patterns, but can result in overfitting if too high.

learning_rate determines the influence of each single tree on the overall model. A low learning rate tends to enhance generalization but demands a higher number of trees in order to result in high accuracy.

Users have these parameters to trade off bias vs. variance, training time, and model complexity. Both are accepted during model initialization:

```python
model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

```
Users can test different values based on dataset size, level of noise, and target accuracy, enabling flexible and optimized learning.


* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Yes, the present implementation is weak at handling noisy, high-dimensional, or complex non-linear data sets. Since it relies on decision stumps (depth-1 trees) as individual base learners, the model will be weak to learn complex feature interactions or decision boundaries in such data sets. Further, since the model at present supports binary classification alone, it cannot be directly applied to multi-class problems without augmentation.

There is also no built-in mechanism such as early stopping, tree depth limitation, or regularization as would be typical in more advanced libraries for gradient boosting. Furthermore, the model is not yet capable of handling missing values or categorical features by default.

These are not the inherent limitations of the boosting algorithm itself, but areas where implementation can be perfected with more time. Becoming better at handling deeper trees, multi-classification, and stronger regularization schemes would make the model more scalable and suitable for more real-world applications.
