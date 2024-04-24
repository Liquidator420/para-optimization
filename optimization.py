import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import numpy as np

# Load the CSV file into a DataFrame
file_path = "D:/study/data science/predictive analysis/para-optimization/agepred.csv"
data = pd.read_csv(file_path)

# Create a DataFrame to store the results
X = data[['SEQN', 'RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']]
y = data['RIDAGEYR']

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Sample', 'Best_C', 'Best_Epsilon', 'Best_Kernel', 'Best_Accuracy'])

# Repeat the process for 10 different samples
for sample_num in range(10):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    best_accuracy = -1
    best_C = None
    best_epsilon = None
    best_kernel = None
    
    # Run 100 iterations for optimization
    for i in range(100):
        # Randomly select parameters for SVM
        C = np.random.uniform(0.1, 10.0)  # Random C value between 0.1 and 10.0
        epsilon = np.random.uniform(0.01, 0.1)  # Random epsilon value between 0.01 and 0.1
        kernel = np.random.choice(['linear', 'poly', 'rbf'])  # Randomly select kernel type
        
        # Train SVR model with selected parameters
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
        svr.fit(X_train, y_train)
        
        # Evaluate model on test set
        predictions = svr.predict(X_test)
        accuracy = r2_score(y_test, predictions)
        
        # Update best parameters if accuracy improves
        if best_accuracy == -1 or accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_epsilon = epsilon
            best_kernel = kernel
        print(sample_num)
        print(i)
    # Save best parameters and accuracy in results DataFrame
    results_df = results_df.append({sample_num + 1,
                                    best_C,
                                    best_epsilon,
                                    best_kernel,
                                    best_accuracy}, ignore_index=True)

# Print the results
print(results_df)