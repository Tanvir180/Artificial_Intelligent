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

df = pd.read_csv('/home/tanvir/ML_Exam/CNN/Dataset1OH/sentiment_tweets3.csv', encoding='ISO-8859-1', low_memory=False)
df = df.rename(columns={"message to examine": "text", "label (depression result)": "label"})
df = df[["text", "label"]]

# Tokenize and pad sequences
max_words = 10000  # Adjust based on your dataset size
max_len = 100  # Adjust based on your sequence length
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(df['label'])

# Build the model without hyperparameter tuning
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, (y > 0).astype(int), test_size=0.2, random_state=42)

# Train the model
epochs = 5  # Adjust as needed
batch_size = 64  # Adjust as needed
csv_logger = CSVLogger('training_details_cnn.csv')

model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=[csv_logger])

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Model Accuracy on Test Set: {accuracy}')

