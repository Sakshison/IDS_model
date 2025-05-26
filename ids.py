import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# Step 1: Load the dataset
dataset_path = "C:/Users/sonis/OneDrive/Desktop/KDD_99/kddcup.data_10_percent_corrected"
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
                "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
                "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
                "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
                "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
                "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
                "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
                "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
df = pd.read_csv(dataset_path, header=None, names=column_names)

# Step 2: Reduce the dataset size
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)  # Use 10% of the dataset for faster processing

# Step 3: Data preprocessing
encoder = LabelEncoder()
for col in ["protocol_type", "service", "flag", "label"]:
    df[col] = encoder.fit_transform(df[col])

selected_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
                     "count", "srv_count", "serror_rate", "srv_serror_rate", "label"]
df = df[selected_features]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize classifiers
models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', tol=1e-4, max_iter=1000),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

# Step 6: Define epochs
epochs = [50, 100, 200, 500]

# Step 7: Train and evaluate models
accuracy_results = {model: [] for model in models.keys()}
efficiency_results = {model: [] for model in models.keys()}

# Loop through epochs
for epoch in epochs:
    print(f"\nEpoch: {epoch}")
    for name, model in models.items():
        if name == "Logistic Regression":
            model.set_params(max_iter=epoch)  # Adjust max_iter for Logistic Regression
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        efficiency = 1 / (training_time + prediction_time) if (training_time + prediction_time) > 0 else 0
        
        # Store results
        accuracy_results[name].append(accuracy)
        efficiency_results[name].append(efficiency)

        # Print results in terminal
        print(f"{name}: Accuracy = {accuracy:.4f}, Efficiency = {efficiency:.4f}")

# Step 8: Plot Accuracy vs Epochs
plt.figure(figsize=(12, 6))
for name, accuracies in accuracy_results.items():
    plt.plot(epochs, accuracies, marker='o', label=name)

plt.title("Accuracy vs Epochs for Different Algorithms")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Plot Efficiency vs Epochs
plt.figure(figsize=(12, 6))
for name, efficiencies in efficiency_results.items():
    plt.plot(epochs, efficiencies, marker='o', label=name)

plt.title("Efficiency vs Epochs for Different Algorithms")
plt.xlabel("Epochs")
plt.ylabel("Efficiency")
plt.legend()
plt.grid(True)
plt.show()
