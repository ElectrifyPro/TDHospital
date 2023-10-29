import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Function to skip over cells with empty or ambiguous data

def data_preprocessing(df, death=False):
    col_to_keep = ['timeknown', 'age', 'psych2', 'information']
    if death:
        col_to_keep = ['death'] + col_to_keep

    df = df[col_to_keep]
    df.shape

    # AGE PROCESSING
    df['age'] = df['age'].apply(lambda x: np.nan if float(x) > 120 else float(x))

    columns_to_process = ['timeknown', 'age', 'psych2', 'information']
    for col in columns_to_process:
        # Calculate mean and standard deviation for the current column
        col_mean = df[col].mean()
        col_std = df[col].std()

        # Define the lower and upper thresholds for outliers
        lower_threshold = col_mean - 3 * col_std
        upper_threshold = col_mean + 3 * col_std

        # Replace values outside the threshold with NaN
        df[col] = df[col].apply(lambda x: col_mean if x < lower_threshold or x > upper_threshold else x)


    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)

    # df['sex'] = df['sex'].apply(lambda value: 1 if value.lower()[0] == 'm' else 0)
    return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)

    # Define the neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layerlayers.Dense(1024, activation='relu'),     # Hidden layer with 256 neurons and ReLU activation
        layers.Dense(512, activation='relu'),     # Hidden layer with 256 neurons and ReLU activation
        layers.Dense(256, activation='relu'),     # Hidden layer with 256 neurons and ReLU activation
        layers.Dense(128, activation='relu'),     # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(64, activation='relu'),      # Another hidden layer with 64 neurons and ReLU activation
        layers.Dense(1, activation='sigmoid')     # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    model.save('example.h5')
    
    print(f'Test accuracy: {test_accuracy}')

    # Optionally, you can plot training history to visualize model performance
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()



if __name__ == "__main__":
    data_path = './TD_HOSPITAL_TRAIN_modified.csv'
    df = pd.read_csv(data_path)
    print("Original data:")
    print(df)

    cleaned_data = data_preprocessing(df, death=True)
    print("Cleaned data:")
    print(cleaned_data)
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)
    train_model(X, y)
    