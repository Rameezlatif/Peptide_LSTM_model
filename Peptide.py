
Peptide
/NLRP3_LSTM_model.ipynb

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Load the peptide sequences from a .txt file
with open("sequences.txt", "r") as f:
    sequences = f.readlines()

# Encode the sequences as one-hot vectors
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)
encoded_sequences = tokenizer.texts_to_matrix(sequences, mode="binary")

# Split the data into train and test sets
train_size = int(len(encoded_sequences) * 0.8)
X_train, X_test = encoded_sequences[:train_size], encoded_sequences[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(encoded_sequences.shape[1],)))
model.add(Dense(encoded_sequences.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, X_train, batch_size=128, epochs=50, validation_data=(X_test, X_test))
# Define the seed sequence
seed_sequence = 'MVHLTPEEK'
# Encode the seed sequence as a one-hot vector
seed_sequence = np.array([char_to_int[c] for c in seed_sequence])
seed_sequence = np_utils.to_categorical(seed_sequence, num_classes=num_classes)
seed_sequence = seed_sequence.reshape(1, seed_sequence.shape[0], seed_sequence.shape[1])

# Generate new sequences
generated_sequence = seed_sequence
for i in range(10):
    # Predict the next character
    next_char_probs = model.predict(generated_sequence)[0,-1,:]
    # Sample the next character
    next_char = np.random.choice(range(num_classes), p=next_char_probs)
    # Append the next character to the generated sequence
    generated_sequence = np.concatenate((generated_sequence, np_utils.to_categorical(next_char, num_classes=num_classes).reshape(1,1,num_classes)))

# Decode the generated sequence
generated_sequence = np.argmax(generated_sequence, axis=-1)
generated_sequence = ''.join([int_to_char[c] for c in generated_sequence[0]])
print(generated_sequence)
