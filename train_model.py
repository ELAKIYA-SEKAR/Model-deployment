# train_model.py
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load preprocessed data
with open('data.pkl', 'rb') as f:
    X, Y, labelEncoder = pickle.load(f)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Train the Decision Tree model
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

# Evaluate the model
y_pred_gini = dt_clf_gini.predict(X_test)
print("Decision Tree using Gini Index\nAccuracy is", accuracy_score(y_test, y_pred_gini) * 100)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(dt_clf_gini, f)
