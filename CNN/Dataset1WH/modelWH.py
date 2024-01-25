import keras_tuner
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback, CSVLogger
from kerastuner.tuners import RandomSearch


df = pd.read_csv('/home/tanvir/ML_Exam/CNN/DatasetT/sentiment_tweets3.csv', encoding =('ISO-8859-1'),low_memory =False)
#df['class'] = df['class'].replace('non-suicide', 0)
#df['class'] = df['class'].replace('suicide', 1)
df = df.rename(columns = {"message to examine":"text", "label (depression result)" :"label" })
df = df[["text", "label"]]

# Tokenize and pad sequences
max_words = 10000  # Adjust based on your dataset size
max_len = 100  # Adjust based on your sequence length
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(df['label'])

# Function to build the model
def build_cnn_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]))
    model.add(Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=16),
                     kernel_size=hp.Int('conv_kernel_size', min_value=3, max_value=7),
                     activation='relu'))
                     
    # Add tunable dropout layer
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )

    return model

# Split the dataset
X_trainn, X_testt, y_trainn, y_testt = train_test_split(X, (y > 0).astype(int), test_size=0.2, random_state=42)

with open('output_cnnhyper.txt', 'w') as f:
    pass  # This line clears the file
# Define a callback to log output to a file
class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('output_cnnhyper.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1} - loss: {logs["loss"]}, accuracy: {logs["accuracy"]}, '
                    f'val_loss: {logs["val_loss"]}, val_accuracy: {logs["val_accuracy"]}\n')

# Add CSVLogger callback to log training details
csv_logger = CSVLogger('training_details_hyper_cnn.csv')

# Instantiate the tuner
cnn_tuner = RandomSearch(
    build_cnn_model,
    objective='val_accuracy',
    max_trials=5,  # Adjust as needed
    project_name='cnn_hyperparameter_tuning'
)

# Perform hyperparameter tuning
cnn_tuner.search(X_trainn, y_trainn, epochs=5, validation_data=(X_testt, y_testt), batch_size=64, callbacks=[EpochLogger(), csv_logger])

# Get the best CNN model
best_cnn_model = cnn_tuner.get_best_models(num_models=1)[0]

# Evaluate the best CNN model
y_pred_cnn = best_cnn_model.predict(X_testt)
y_pred_cnn = (y_pred_cnn > 0.5).astype(int)
accuracy_cnn = accuracy_score(y_testt, y_pred_cnn)
print(f'Best CNN Model Accuracy: {accuracy_cnn}')

# Get the best hyperparameters for CNN
best_cnn_hyperparameters = cnn_tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
print(f'Best CNN Hyperparameters: {best_cnn_hyperparameters}')

# Add the following lines to write the output to a log file named 'output_log.txt'
with open('output_cnnhyper.txt', 'a') as f:
    f.write(f'Best CNN Model Accuracy: {accuracy_cnn}\n')
    f.write(f'Best CNN Hyperparameters: {best_cnn_hyperparameters}\n')
