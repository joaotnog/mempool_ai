import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam

# Load the feature-engineered data
transactions_with_proxies = pd.read_csv('../data/transactions_with_proxies.csv')

# Load the labels data
seq_nonce_backtesting_results = pd.read_csv('../data/seq_nonce_backtesting_results.csv')

# Ensure the 'Index' column is present in both dataframes and has the same name
# Rename 'Transaction Hash' to 'hash' in the labels data if needed
seq_nonce_backtesting_results.rename(columns={'Transaction Hash': 'hash'}, inplace=True)

# Merge the datasets on the 'Index' column
merged_data = pd.merge(transactions_with_proxies, seq_nonce_backtesting_results, on='hash', how='inner')

# Features and labels
features = [
    'dumb_flow_proxy', 'urgency_proxy', 'transaction_burstiness_proxy',
    'gas_price_volatility_proxy', 'large_transaction_impact_proxy',
    'rolling_avg_gas_price', 'rolling_std_gas_price', 'rolling_avg_value',
    'transaction_count'
]
X = merged_data[features]
X.fillna(0, inplace=True)
y = merged_data['Classification'].map({'Loss': 0, 'Profitable': 1})

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM [samples, timesteps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001, mode='min')

# Define the model building function for Keras Tuner
def build_model(hp):
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    attention = MultiHeadAttention(
        num_heads=hp.Choice('num_heads', values=[2, 4]),
        key_dim=hp.Choice('key_dim', values=[2, 4])
    )(input_layer, input_layer)
    
    norm = LayerNormalization(epsilon=1e-6)(attention)
    
    dense = Dense(
        units=hp.Choice('units', values=[32, 64]),
        activation='relu'
    )(norm)
    
    dropout = Dropout(
        rate=hp.Choice('dropout', values=[0.2, 0.3])
    )(dense)
    
    flatten = Flatten()(dropout)
    
    output_layer = Dense(1, activation='sigmoid')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(
        optimizer=Adam(
            hp.Choice('learning_rate', values=[1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    hyperband_iterations=1,
    directory='tuning_results',
    project_name='transformer_tuning'
)

# Search for best hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
                         callbacks=[early_stopping, reduce_lr], verbose=2)

# Evaluate the model
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the model
best_model.save('models/best_transformer_mev_model.h5')
