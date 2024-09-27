import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create a synthetic dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

def stepwise_regression(X, y, threshold_in=0.01, threshold_out=0.05):
    initial_features = X.columns.tolist()
    selected_features = []
    
    while True:
        changed = False
        
        # Forward step
        remaining_features = list(set(initial_features) - set(selected_features))
        best_pval = 1
        best_feature = None
        
        for feature in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
            pval = model.pvalues[feature]
            if pval < best_pval:
                best_pval = pval
                best_feature = feature
        
        if best_pval < threshold_in:
            selected_features.append(best_feature)
            changed = True
        
        # Backward step
        model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
        for feature in selected_features:
            pval = model.pvalues[feature]
            if pval > threshold_out:
                selected_features.remove(feature)
                changed = True
        
        if not changed:
            break
    
    return selected_features

# Define features and target variable
X = df.drop(columns='target')
y = df['target']

# Perform stepwise regression
selected_features = stepwise_regression(X, y)

# Fit the final model using the selected features
final_model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

# Output the summary of the final model
print(final_model.summary())

# Predictions
predictions = final_model.predict(sm.add_constant(X[selected_features]))

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(y, predictions))
mae = mean_absolute_error(y, predictions)

print(f"Selected features: {selected_features}")
print(f"R-squared: {final_model.rsquared:.4f}")
print(f"Adjusted R-squared: {final_model.rsquared_adj:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
