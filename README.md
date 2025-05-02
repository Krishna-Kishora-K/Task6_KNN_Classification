# Task 6: K-Nearest Neighbors (KNN) Classification

## Objective
To build a K-Nearest Neighbors (KNN) classifier using the built-in Iris dataset and evaluate its performance for different values of K.

---

## Tools & Libraries Used

- Python
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

## Dataset

- **Dataset:** Iris Dataset (built-in)
- **Source:** Automatically loaded from `sklearn.datasets`
- **Classes:** Setosa, Versicolor, Virginica
- **Features:** Sepal length, Sepal width, Petal length, Petal width

---

## Steps Performed

1. **Loaded** the Iris dataset using `sklearn.datasets.load_iris()`
2. **Normalized** the features using `StandardScaler` to improve KNN performance
3. **Split** the data into training and testing sets (80/20)
4. **Trained** KNN models with different values of `k` (from 1 to 10)
5. **Evaluated** models using accuracy, confusion matrix, and classification report
6. **Plotted** the decision boundary using the first two features
7. **Selected** the best value of K based on test accuracy

---

## Best Result

- **Best K:** 1  
- **Accuracy:** 100%  
- **Confusion Matrix:**
- [[10 0 0]
  [ 0 7 0]
  [ 0 0 13]]

