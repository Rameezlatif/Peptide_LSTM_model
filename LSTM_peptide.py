import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Sample sequences (you can replace this with your dataset)
sequences = [
    
    "NQPVESDES",
        # ... (your sequences here)
]

# Tokenize input sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)
total_chars = len(tokenizer.word_index) + 1

# Generate input sequences and labels
input_sequences = []
labels = []
max_sequence_length = 9  # Maximum length of generated sequences

for seq in sequences:
    for i in range(1, len(seq)):
        input_seq = seq[:i]
        label = seq[i]
        input_sequences.append(input_seq)
        labels.append(label)

# Convert sequences to numerical data
input_sequences = tokenizer.texts_to_sequences(input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
labels = tokenizer.texts_to_sequences(labels)
labels = np.array(labels).reshape(-1,)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(max_sequence_length, 1)))
model.add(Dense(total_chars, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(input_sequences, labels, epochs=100, batch_size=32)

# Function to generate novel sequences with a timeout mechanism
def generate_sequences_with_timeout(model, tokenizer, max_length, seed_text, num_sequences=10, timeout=10):
    start_time = time.time()
    generated_sequences = set()
    while len(generated_sequences) < num_sequences and time.time() - start_time < timeout:
        generated_sequence = seed_text
        for _ in range(max_length):
            encoded_text = tokenizer.texts_to_sequences([generated_sequence])[0]
            encoded_text = pad_sequences([encoded_text], maxlen=max_length, padding='pre')
            encoded_text = np.reshape(encoded_text, (1, max_length, 1))
            predicted_char_index = np.argmax(model.predict(encoded_text), axis=1)[0]
            predicted_char = tokenizer.index_word[predicted_char_index]
            generated_sequence += predicted_char
        # Check if the generated sequence is unique and not equal to the seed sequence
        if generated_sequence != seed_text and generated_sequence not in generated_sequences:
            generated_sequences.add(generated_sequence)
    return list(generated_sequences)

# Define max_length before generating sequences
max_length = 9  # Length of the generated sequence

# Generate at least 10 novel sequences with a timeout of 10 seconds for each sequence
seed_text = "Q"  # Seed sequence of length 1
num_generated_sequences = 10  # At least 10 sequences
timeout_per_sequence = 10  # Timeout for each sequence in seconds
generated_sequences = generate_sequences_with_timeout(model, tokenizer, max_length, seed_text, num_generated_sequences, timeout_per_sequence)

# Print and save the generated novel sequences to a text file
output_file_path = "generated_sequences.txt"
with open(output_file_path, 'w') as output_file:
    for idx, seq in enumerate(generated_sequences):
        print(f"Novel Sequence {idx + 1}: {seq}")
        output_file.write(f"Novel Sequence {idx + 1}: {seq}\n")

print(f"Generated sequences saved to: {output_file_path}")

