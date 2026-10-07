# IMDb Sentiment Analysis using LSTM

This project implements **Sentiment Analysis on IMDb movie reviews using an LSTM (Long Short-Term Memory) neural network** with TensorFlow/Keras.

The model takes a movie review as a sequence of word IDs and predicts whether the review is **positive or negative**.

---

## 🚀 Project Overview

The complete workflow of this project is:

```text
IMDb Reviews
     ↓
Load Dataset
     ↓
Keep Top 5000 Words
     ↓
Integer-Encoded Reviews
     ↓
Padding to 400 Words
     ↓
Embedding Layer
     ↓
LSTM Layer
     ↓
Dense + Sigmoid
     ↓
Positive / Negative Prediction
```

---

## 📂 Dataset

The project uses the built-in **IMDb Movie Reviews dataset** provided by Keras.

The dataset contains:

* **25,000 training reviews**
* **25,000 testing reviews**
* Binary sentiment labels:

  * `0` → Negative
  * `1` → Positive

Only the **top 5,000 most frequent words** are used.

```python
vocab_size = 5000
```

---

## 🧠 Model Architecture

The model consists of three main layers:

### 1. Embedding Layer

```python
Embedding(
    input_dim=5000,
    output_dim=32
)
```

Converts each word ID into a **32-dimensional dense vector**.

```text
Word ID
   ↓
Embedding
   ↓
32-dimensional vector
```

---

### 2. LSTM Layer

```python
LSTM(
    128,
    activation="tanh",
    return_sequences=False
)
```

The LSTM processes the review sequence and learns important information from previous words.

It uses **128 LSTM units** to capture patterns in the text.

`return_sequences=False` means the LSTM returns only the **final output** of the sequence.

---

### 3. Output Layer

```python
Dense(
    1,
    activation="sigmoid"
)
```

The sigmoid function produces a value between `0` and `1`.

```text
0 → Negative Review
1 → Positive Review
```

---

## 📊 Data Preprocessing

### Vocabulary Limitation

Only the top 5,000 frequently occurring words are kept:

```python
imdb.load_data(num_words=5000)
```

### Padding

Reviews have different lengths, so they are converted to a fixed length of **400 words**:

```python
max_words = 400

x_train = sequence.pad_sequences(
    x_train,
    maxlen=max_words
)
```

This makes the input shape:

```text
Training Data → (24936, 400)
Test Data     → (25000, 400)
```

---

## 🔀 Validation Set

64 samples are separated from the training data for validation:

```python
x_valid = x_train[:64]
y_valid = y_train[:64]

x_train = x_train[64:]
y_train = y_train[64:]
```

The validation set is used to monitor model performance during training.

---

## ⚙️ Training Configuration

The model is compiled using:

```python
LSTM_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

### Parameters

| Parameter             |               Value |
| --------------------- | ------------------: |
| Vocabulary Size       |                5000 |
| Maximum Review Length |                 400 |
| Embedding Dimension   |                  32 |
| LSTM Units            |                 128 |
| Batch Size            |                  64 |
| Epochs                |                   5 |
| Optimizer             |                Adam |
| Loss Function         | Binary Crossentropy |
| Output Activation     |             Sigmoid |

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* LSTM
* Natural Language Processing (NLP)
* IMDb Dataset

---

## 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Install the required dependencies:

```bash
pip install tensorflow numpy
```

---

## ▶️ Run the Project

Run the Python file:

```bash
python main.py
```

The program will:

1. Load the IMDb dataset
2. Display a review as integer IDs
3. Convert IDs back to words
4. Check review lengths
5. Pad reviews to 400 words
6. Create a validation set
7. Build the LSTM model
8. Compile the model
9. Train the model for 5 epochs

---

## 🔍 Why LSTM?

Traditional RNNs can struggle to remember information over long sequences because of the **vanishing gradient problem**.

LSTM solves this using special gates that control information flow:

```text
Input Gate
    ↓
Forget Gate → Cell State → Output Gate
    ↓
Hidden State
```

This makes LSTM useful for sequence-based tasks such as:

* Sentiment Analysis
* Text Classification
* Language Modeling
* Sequence Prediction
* Time-Series Prediction

---

## 📈 Expected Result

After training, the model produces training and validation metrics such as:

```text
Epoch 1/5
...
accuracy: ...

Epoch 2/5
...
accuracy: ...

...

Epoch 5/5
...
accuracy: ...
```

The exact accuracy can vary depending on the TensorFlow/Keras version and training environment.

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Text preprocessing
* Tokenized/encoded text sequences
* Vocabulary limitation
* Padding sequences
* Word embeddings
* LSTM architecture
* Binary classification
* Sigmoid activation
* Binary cross-entropy
* Adam optimizer
* Training and validation
* Sentiment analysis

---

## 👨‍💻 Author

**Sourabh Sharma**

AI / ML / Generative AI Enthusiast

---

## ⭐ Future Improvements

* Add test-set evaluation
* Add confusion matrix
* Add accuracy/loss plots
* Build a prediction function for custom reviews
* Deploy the model using FastAPI or Streamlit
* Compare LSTM with SimpleRNN and GRU
