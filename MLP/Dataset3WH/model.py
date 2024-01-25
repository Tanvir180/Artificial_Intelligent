import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.layers import Embedding, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback, CSVLogger
from kerastuner.tuners import RandomSearch

# Load CSV file
df = pd.read_csv('/home/tanvir/ML_Exam/CNN/Dataset3OH/IMDB Dataset.csv', encoding='ISO-8859-1', low_memory=False)
df = df.rename(columns={"review": "text", "sentiment": "label"})
df['label'] = df['label'].replace('negative', 0)
df['label'] = df['label'].replace('positive', 1)
df = df[["text", "label"]]

# Tokenize and pad sequences
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(df['label'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, (y > 0).astype(int), test_size=0.2, random_state=42)

## Function to build the MLP model
def build_mlp_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=hp.Int('embedding_dim', min_value=32, max_value=256, step=32), input_length=max_len))
    model.add(Flatten())
    model.add(Dense(hp.Int('units', min_value=64, max_value=256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )

    return model

# Clear the file and write the output to 'output_mlp_hyper.txt'
with open('output_mlp_hyper.txt', 'w') as f:
    pass

# Define a callback to log output to a file
class EpochLoggerMLP(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('output_mlp_hyper.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1} - loss: {logs["loss"]}, accuracy: {logs["accuracy"]}, '
                    f'val_loss: {logs["val_loss"]}, val_accuracy: {logs["val_accuracy"]}\n')

# Add CSVLogger callback to log training details
csv_logger_mlp = CSVLogger('training_details_mlp_hyper.csv')

# Instantiate the MLP tuner
mlp_tuner = RandomSearch(
    build_mlp_model,
    objective='val_accuracy',
    max_trials=5,  # Adjust as needed
    project_name='mlp_hyperparameter_tuning'
)

# Perform hyperparameter tuning for the MLP model
mlp_tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64,
                  callbacks=[EpochLoggerMLP(), csv_logger_mlp])

# Get the best MLP model
best_mlp_model = mlp_tuner.get_best_models(num_models=1)[0]

# Evaluate the best MLP model
y_pred_mlp = best_mlp_model.predict(X_test)
y_pred_mlp = (y_pred_mlp > 0.5).astype(int)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f'Best MLP Model Accuracy: {accuracy_mlp}')

# Get the best hyperparameters for MLP
best_mlp_hyperparameters = mlp_tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
print(f'Best MLP Hyperparameters: {best_mlp_hyperparameters}')

# Write the output to a log file named 'output_mlp_hyper.txt'
with open('output_mlp_hyper.txt', 'a') as f:
    f.write(f'Best MLP Model Accuracy: {accuracy_mlp}\n')
    f.write(f'Best MLP Hyperparameters: {best_mlp_hyperparameters}\n')
