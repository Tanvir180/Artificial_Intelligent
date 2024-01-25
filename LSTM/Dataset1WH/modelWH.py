import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback, CSVLogger
from kerastuner.tuners import RandomSearch

# Load dataset
df = pd.read_csv('/home/tanvir/ML_Exam/LSTM/Dataset1WH/sentiment_tweets3.csv', encoding='ISO-8859-1', low_memory=False)
df = df.rename(columns={"message to examine": "text", "label (depression result)": "label"})
df = df[["text", "label"]]

# Tokenize and pad sequences
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(df['label'])

# Function to build the LSTM model
def build_lstm_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]))
    model.add(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=16), return_sequences=True))
    
    # Add tunable dropout layer
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=16)))
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )

    return model

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, (y > 0).astype(int), test_size=0.2, random_state=42)

# Define a callback to log output to a file
class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('output_lstm_hyper.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1} - loss: {logs["loss"]}, accuracy: {logs["accuracy"]}, '
                    f'val_loss: {logs["val_loss"]}, val_accuracy: {logs["val_accuracy"]}\n')

# Add CSVLogger callback to log training details
csv_logger = CSVLogger('training_details_hyper_lstm.csv')

# Instantiate the tuner
lstm_tuner = RandomSearch(
    build_lstm_model,
    objective='val_accuracy',
    max_trials=5,  # Adjust as needed
    project_name='lstm_hyperparameter_tuning'
)

# Perform hyperparameter tuning
lstm_tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64, callbacks=[EpochLogger(), csv_logger])

# Get the best LSTM model
best_lstm_model = lstm_tuner.get_best_models(num_models=1)[0]

# Evaluate the best LSTM model
y_pred_lstm = best_lstm_model.predict(X_test)
y_pred_lstm = (y_pred_lstm > 0.5).astype(int)
accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
print(f'Best LSTM Model Accuracy: {accuracy_lstm}')

# Get the best hyperparameters for LSTM
best_lstm_hyperparameters = lstm_tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
print(f'Best LSTM Hyperparameters: {best_lstm_hyperparameters}')

# Add the following lines to write the output to a log file named 'output_log.txt'
with open('output_lstm_hyper.txt', 'a') as f:
    f.write(f'Best LSTM Model Accuracy: {accuracy_lstm}\n')
    f.write(f'Best LSTM Hyperparameters: {best_lstm_hyperparameters}\n')

