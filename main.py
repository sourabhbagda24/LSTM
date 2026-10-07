from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

import numpy as np


# ============================================================
# 2. Load IMDb Dataset
# ============================================================

# Keep only the top 5000 most frequent words
vocab_size = 5000

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))


# ============================================================
# 3. See One Review as Numbers
# ============================================================

print("\nFirst review as integers:")
print(x_train[0])


# ============================================================
# 4. Convert Integer IDs to Words
# ============================================================

word_idx = imdb.get_word_index()

# Reverse the dictionary
word_idx = {i: word for word, i in word_idx.items()}

print("\nFirst review as words:")
print([word_idx[i] for i in x_train[0] if i in word_idx])


# ============================================================
# 5. Check Review Length
# ============================================================

print(
    "\nMaximum review length:",
    len(max((x_train + x_test), key=len))
)

print(
    "Minimum review length:",
    len(min((x_train + x_test), key=len))
)


# ============================================================
# 6. Make All Reviews Same Length
# ============================================================

# Fixed length for every review
max_words = 400

x_train = sequence.pad_sequences(
    x_train,
    maxlen=max_words
)

x_test = sequence.pad_sequences(
    x_test,
    maxlen=max_words
)

print("\nShape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)


# ============================================================
# 7. Create Validation Set
# ============================================================

x_valid = x_train[:64]
y_valid = y_train[:64]

x_train = x_train[64:]
y_train = y_train[64:]

print("\nTraining data after validation split:")
print(x_train.shape)

print("Validation data:")
print(x_valid.shape)


# ============================================================
# 8. Embedding Size
# ============================================================

embd_len = 32


# ============================================================
# 9. Create LSTM Model
# ============================================================

LSTM_model = Sequential(name="LSTM")


# Convert word IDs into dense vectors
LSTM_model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embd_len
    )
)


# ============================================================
# 10. LSTM Layer
# ============================================================

LSTM_model.add(
    LSTM(
        128,
        activation="tanh",
        return_sequences=False
    )
)


# ============================================================
# Output Layer
# ============================================================

LSTM_model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ============================================================
# 11. Compile Model
# ============================================================

LSTM_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# ============================================================
# 12. Train Model
# ============================================================

history = LSTM_model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(x_valid, y_valid)
)