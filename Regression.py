# https://data.mendeley.com/datasets/8v757rr4st/1/files/98242fd3-1912-4a59-ab26-23d97b454218

# Use dataset from above website randomly split it into two paritions, try to use 10 cross-validations( or any other better method you know to maximize the output!!)

# We will be using the training dataset to train this ML model and use regression on same training dataset to get weights for different soil features, then we will use derrived wieghts to model our soil bio sensor.
# We need to use complex regression methods to capture more complexities of data along with that use regularization to avoid overfitting 

# Then we will have final showdown between these two models, on testing dataset. 

# Final Score will be given taking ML model as the base.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

filename = "Crop Recommendation using Soil Properties and Weather Prediction.csv"
df = pd.read_csv(filename)

X = df.drop(columns=['label'])
y = df['label']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

encoded_cat_cols = []
if categorical_cols:
    encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols).tolist()

final_feature_names = numeric_cols + encoded_cat_cols
n_features = len(final_feature_names)

np.random.seed(42)
n_sensors = 8

rows = []
for crop in sorted(y.unique()):
    X_crop = X_processed[y == crop]
    true_W_crop = np.random.randn(n_features, n_sensors) * 0.5
    Y_sensor_crop = np.tanh(X_crop @ true_W_crop) + 0.05 * np.random.randn(* (X_crop @ true_W_crop).shape)
    ridge = Ridge(alpha=0.1, fit_intercept=False)
    ridge.fit(X_crop, Y_sensor_crop)
    W_est_crop = ridge.coef_.T   
  
    flat_weights = W_est_crop.flatten(order='F')  
  
    row_dict = {'Crop': crop}
    for i, feature in enumerate(final_feature_names):
        for j in range(n_sensors):
            row_dict[f'{feature}_S{j+1}'] = W_est_crop[i, j]
    
    rows.append(row_dict)


weights_df = pd.DataFrame(rows)
output_file = "Cropwise_Biosensor_Weights.csv"
weights_df.to_csv(output_file, index=False)

print(f"\n File saved: {output_file}")
print(f" Shape: {weights_df.shape}")
print("Columns sample:", weights_df.columns[:10].tolist())
print(weights_df.head(3))


