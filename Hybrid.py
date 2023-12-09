import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset
data = pd.read_csv("BostonWeather.csv")

# Data preprocessing for ARIMA
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)  # Set 'date' column as the index
data = data.resample('D').mean()
data = data.fillna(method='ffill')

# Fit GRU model
data_gru = data.copy()

# Initialize two scalers: one for features and one for the target variable
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Feature columns and target variable
feature_columns = ['tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']  # Adjust this list as per your dataset
target_column = 'tavg'

# Normalize feature columns
data_gru[feature_columns] = feature_scaler.fit_transform(data_gru[feature_columns])

# Normalize target column
data_gru[target_column] = target_scaler.fit_transform(data_gru[[target_column]])

# Function to create sequences with multiple features
def create_sequences(data_gru, sequence_length, feature_columns, target_column):
    X = []
    y = []
    for i in range(len(data_gru) - sequence_length):
        X.append(data_gru[feature_columns].iloc[i:i+sequence_length].values)
        y.append(data_gru[target_column].iloc[i + sequence_length])
    return np.array(X), np.array(y)

# Define sequence length and create sequences
sequence_length = 10
X, y = create_sequences(data_gru, sequence_length, feature_columns, target_column)

# Splitting data into training, validation, and test sets
train_size = int(0.70 * len(X))  # 70% for training
val_size = int(0.10 * len(X))   # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 20% for test

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# Reshape sequences for GRU input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(feature_columns)))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], len(feature_columns)))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(feature_columns)))

# Define hyperparameters for tuning
layer_sizes = [16, 32, 64]  # Different sizes for GRU layers
layer_numbers = [2, 3, 4]  # Different numbers of GRU layers
batch_sizes = [16, 32, 64] # Different batch sizes
learning_rates = [0.001, 0.01, 0.1]  # Different learning rates

best_loss = float('inf')
best_model = None
best_params = {}
validation_losses = []  # To store validation losses for different configurations

for size in layer_sizes:
    for num in layer_numbers:
        for batch in batch_sizes:
            for lrs in learning_rates:
                print(f"Training with Size: {size}, Num: {num}, Batch: {batch}, LR: {lrs}")

                model = Sequential()
                model.add(GRU(size, activation='relu', return_sequences=True, input_shape=(sequence_length, len(feature_columns))))
                for _ in range(num - 1):
                    model.add(GRU(size, activation='relu', return_sequences=True))
                model.add(GRU(size, activation='relu'))
                model.add(Dense(1))

                model.compile(optimizer=Adam(lr=lrs), loss='mean_squared_error')

                history = model.fit(X_train, y_train, epochs=10, batch_size=batch, validation_data=(X_val, y_val), verbose=0)
                
                # Store validation loss for each configuration
                val_loss = history.history['val_loss'][-1]
                validation_losses.append((size, num, lrs, batch, val_loss))

                # Evaluate model
                val_loss = history.history['val_loss'][-1]
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model
                    best_params = {'Size': size, 'Num': num, 'Batch': batch, 'LR': lrs}

# Sort the validation_losses based on the validation loss in ascending order
sorted_losses = sorted(validation_losses, key=lambda x: x[4])[:10]  # Choose the top 10 configurations

# Extracting hyperparameters from the sorted losses
sizes = [size for size, num, lrs, batch, loss in sorted_losses]
numbers = [num for size, num, lrs, batch, loss in sorted_losses]
batches = [batch for size, num, lrs, batch, loss in sorted_losses]
learning_rates = [lrs for size, num, lrs, batch, loss in sorted_losses]

# Plotting top 10 best hyperparameter combinations with improved label visibility
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

# Create horizontal bar plot
plt.barh(np.arange(10), [loss for size, num, lrs, batch, loss in sorted_losses], align='center')

# Set y-axis ticks and labels with proper formatting
yticks_labels = [f'Size:{size}, Num:{num}, LR:{lrs}, Batch:{batch}' for size, num, lrs, batch, loss in sorted_losses]
plt.yticks(np.arange(10), yticks_labels)

# Add labels and title
plt.xlabel('Validation Loss')
plt.title('Top 10 Best Hyperparameter Combinations')
plt.gca().invert_yaxis()

# Adjust layout for better visibility
plt.tight_layout()

# Show the plot
plt.show()

print("Best parameters:", best_params)

best_model = Sequential()
best_model.add(GRU(best_params['Size'], activation='relu', return_sequences=True, input_shape=(sequence_length, len(feature_columns))))
for _ in range(best_params['Num'] - 1):
    best_model.add(GRU(best_params['Size'], activation='relu', return_sequences=True))
best_model.add(GRU(best_params['Size'], activation='relu'))
best_model.add(Dense(1))

# Training the best GRU model
best_model.compile(optimizer=Adam(lr=best_params['LR']), loss='mean_squared_error')
history = best_model.fit(X_train, y_train, epochs=10, batch_size=best_params['Batch'], validation_data=(X_val, y_val), verbose=1)

# Plotting training and validation losses across epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Calculating R-squared for training and validation sets across epochs
r2_train = [r2_score(y_train, best_model.predict(X_train))]
r2_val = [r2_score(y_val, best_model.predict(X_val))]

for epoch in range(1, 10):  # Assuming 10 epochs
    best_model.fit(X_train, y_train, epochs=1, batch_size=best_params['Batch'], verbose=0)
    r2_train.append(r2_score(y_train, best_model.predict(X_train)))
    r2_val.append(r2_score(y_val, best_model.predict(X_val)))

# Plotting R-squared values for training and validation sets across epochs
plt.figure(figsize=(8, 5))
plt.plot(r2_train, label='Training R-squared')
plt.plot(r2_val, label='Validation R-squared')
plt.title('Training and Validation R-squared')
plt.xlabel('Epochs')
plt.ylabel('R-squared')
plt.legend()
plt.show()


# Get GRU predictions on test set
gru_predictions = best_model.predict(X_test).flatten()

# Adjust the length of y_test to match the length of predictions

gru_predictions = gru_predictions[:len(y_test)]

# Fit ARIMA on residuals (difference between actual and GRU predictions for the test set)
residuals_test = y_test - gru_predictions

# Fit ARIMA model on residuals for the test set
arima_model = ARIMA(residuals_test, order=(1, 1, 1))
arima_results = arima_model.fit()

# Predictions on the test set
gru_predictions_test = best_model.predict(X_test)

# Calculating and displaying evaluation metrics on the test set
gru_predictions_test = target_scaler.inverse_transform(gru_predictions_test)
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

mse_test = mean_squared_error(y_test_inv, gru_predictions_test)
r2_test = r2_score(y_test_inv, gru_predictions_test)

print("\nGRU Model Metrics:")
print(f"Mean Squared Error (MSE): {mse_test}")
print(f"R-Squared (R²): {r2_test}")

# Combine GRU and ARIMA predictions for the test set, Reverse normalization for ARIMA predictions to original scale
hybrid_predictions_test = arima_results.fittedvalues + gru_predictions  # Use fitted values from ARIMA
hybrid_predictions_test = target_scaler.inverse_transform(hybrid_predictions_test.reshape(-1, 1)).flatten()

# Calculate MSE and R-squared for ARIMA model on the test set
mse_hybrid = mean_squared_error(target_scaler.inverse_transform(y_test.reshape(-1, 1)), hybrid_predictions_test)
r_squared_hybrid = r2_score(target_scaler.inverse_transform(y_test.reshape(-1, 1)), hybrid_predictions_test)

print("\nHybrid Model Metrics:")
print(f"Mean Squared Error (MSE):{mse_hybrid}")
print(f"R-squared: {r_squared_hybrid}")

# plot

# Create an array of indices for the test set
indices = pd.date_range(start=data_gru.index[train_size + val_size], periods=len(y_test), freq='D')

# Ensure the lengths match for visualization
if len(indices) > len(gru_predictions):
    indices = indices[:len(gru_predictions)]
elif len(gru_predictions) > len(indices):
    gru_predictions = gru_predictions[:len(indices)]

# Visualize predicted vs actual values for the test set (GRU)
plt.figure(figsize=(10, 6))

# Plotting the actual values for the test set
plt.plot(indices, target_scaler.inverse_transform(y_test[:len(indices)].reshape(-1, 1)), label='Actual')

# Plotting predicted values (GRU) for the test set
plt.plot(indices, target_scaler.inverse_transform(gru_predictions.reshape(-1, 1)), linestyle='--', label='GRU Predicted')

plt.title('GRU Model - Predicted vs Actual Average Temperature')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()

# Visualize predicted vs actual values for the test set (Hybrid Model)
plt.figure(figsize=(10, 6))

# Plotting the actual values
plt.plot(indices, target_scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')

# Plotting predicted values (Hybrid)
plt.plot(indices, hybrid_predictions_test, linestyle='--', label='Hybrid Predicted')

plt.title('Hybrid Model - Predicted vs Actual Average Temperature')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()