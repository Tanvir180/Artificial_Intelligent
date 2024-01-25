import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback, CSVLogger
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Load CSV file
df = pd.read_csv('/home/tanvir/ML_Exam/MLP/Dataset2OH/emails.csv', encoding='ISO-8859-1', low_memory=False)
df = df[["text", "spam"]]
df = df.rename(columns={"text": "text", "spam": "label"})

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

# Build the MLP model
embedding_dim = 50
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Clear the file and write the output to 'output_log.txt'
with open('output_mlp.txt', 'w') as f:
    pass

# Define a callback to log output to a file
class EpochLoggerMLP(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('output_mlp.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1} - loss: {logs["loss"]}, accuracy: {logs["accuracy"]}, '
                    f'val_loss: {logs["val_loss"]}, val_accuracy: {logs["val_accuracy"]}\n')

# Add CSVLogger callback to log training details
csv_logger_simple_mlp = CSVLogger('training_details_simple_mlp.csv')

# Train the non-hyperparameter-tuned MLP model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64,
                      callbacks=[EpochLoggerMLP(), csv_logger_simple_mlp])

# Evaluate the non-hyperparameter-tuned MLP model
y_pred_simple_mlp = model.predict(X_test)
y_pred_simple_mlp = (y_pred_simple_mlp > 0.5).astype(int)
accuracy_simple_mlp = accuracy_score(y_test, y_pred_simple_mlp)
print(f'Non-Hyperparameter Tuned MLP Model Accuracy: {accuracy_simple_mlp}')

# Write the output to a log file named 'output_log.txt'
with open('output_mlp.txt', 'a') as f:
    f.write(f'\nSimple MLP Model Accuracy: {accuracy_simple_mlp}\n')




