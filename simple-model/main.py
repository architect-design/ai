import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# 1. Custom Tokenizer (Character Level)
# ==========================================
class FormatTokenizer:
    def __init__(self):
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def fit(self, raw_text_data):
        """Builds vocabulary from the raw text content of uploaded files."""
        unique_chars = sorted(list(set(raw_text_data)))
        self.char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        print(f"Vocabulary built. Vocab Size: {self.vocab_size}")

    def encode(self, text):
        return [self.char_to_int[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.int_to_char[i] for i in indices])


# ==========================================
# 2. Custom AI Model Architecture (LSTM)
# ==========================================
class SpecificationLearner(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SpecificationLearner, self).__init__()

        # Embedding Layer: Converts character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM Layer: The core engine that learns sequences and structure
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Fully Connected Layer: Maps LSTM output back to vocabulary probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x)

        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Reshape for linear layer
        out = lstm_out.reshape(lstm_out.size(0) * lstm_out.size(1), -1)

        # Final output (logits)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        weight = next(self.parameters()).data
        hidden = (weight.new(1, batch_size, hidden_dim).zero_(),
                  weight.new(1, batch_size, hidden_dim).zero_())
        return hidden


# ==========================================
# 3. Helper: Data Preparation
# ==========================================
def get_training_data(raw_text, tokenizer, seq_length=50):
    """
    Chops the text into overlapping sequences.
    Example: "Hello World" -> 
             Input: "Hello", Target: "ello "
             Input: "ello ", Target: "llo W"
    """
    encoded = tokenizer.encode(raw_text)
    inputs = []
    targets = []

    for i in range(0, len(encoded) - seq_length):
        chunk = encoded[i:i + seq_length + 1]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    return torch.tensor(inputs), torch.tensor(targets)


# ==========================================
# 4. Main Execution
# ==========================================

# --- A. SIMULATE UPLOADED FILES ---
# In a real scenario, you would load your files here.
# Let's simulate a specific format: "SPEC_TYPE_A" with JSON-like structure.
# The model must learn that "id" is always followed by numbers, etc.
file_paths = ["ach1.txt"]
all_text = ""
for path in file_paths:
    with open(path, "r", encoding="utf-8") as f:
        all_text += f.read() + "\n"

training_file_content = all_text * 5  # Repeat to give the model enough data to learn patterns

# --- B. HYPERPARAMETERS ---
embedding_dim = 64
hidden_dim = 128
learning_rate = 0.005
epochs = 100
seq_length = 20  # How many characters the model sees at once

# --- C. PREPARE DATA ---
tokenizer = FormatTokenizer()
tokenizer.fit(training_file_content)

vocab_size = tokenizer.vocab_size
model = SpecificationLearner(vocab_size, embedding_dim, hidden_dim)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

inputs, targets = get_training_data(training_file_content, tokenizer, seq_length)
batch_size = 64  # You can adjust this (e.g., 32, 64, 128)
dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Training Data Shape: {inputs.shape}")

print("Starting Training...")
model.train()

for epoch in range(epochs):
    # Reset hidden state for each epoch?
    # For stateless learning, we init hidden per batch.

    for batch_inputs, batch_targets in loader:
        # 1. Initialize hidden state for THIS specific batch size
        # batch_inputs shape: [64, 20]
        current_batch_size = batch_inputs.size(0)
        hidden = model.init_hidden(current_batch_size)

        optimizer.zero_grad()

        # 2. Forward pass
        output, hidden = model(batch_inputs, hidden)

        # 3. Calculate loss
        loss = criterion(output, batch_targets.view(-1))

        # 4. Backward pass
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

print("Training Complete.")


# ==========================================
# 5. GENERATE / VALIDATE FORMAT (FIXED)
# ==========================================
def generate_format_spec(model, tokenizer, start_str, predict_len=200):
    model.eval()

    # SAFETY CHECK: Ensure start_str characters are in the vocabulary
    # If a char is missing, fallback to the first char of the training data
    valid_start = True
    for char in start_str:
        if char not in tokenizer.char_to_int:
            print(f"Warning: Character '{char}' not in training vocabulary. Using default start.")
            valid_start = False
            break

    if not valid_start:
        # Fallback: use the first character of the training data vocabulary
        # (or you could set start_str = training_file_content[:10])
        start_str = list(tokenizer.char_to_int.keys())[0]

    hidden = model.init_hidden(1)

    # Prime the model with the start string
    input_seq = torch.tensor([tokenizer.encode(start_str)])

    generated_text = start_str

    with torch.no_grad():
        for _ in range(predict_len):
            # Run input through model
            output, hidden = model(input_seq, hidden)

            # Get the character with highest probability
            last_char_logits = output[-1, :].unsqueeze(0)

            # Apply Softmax to get probabilities
            probs = torch.softmax(last_char_logits, dim=1)

            # Sample from distribution
            top_idx = torch.multinomial(probs, 1).item()

            # Decode and append
            char = tokenizer.decode([top_idx])
            generated_text += char

            # Prepare next input
            input_seq = torch.tensor([[top_idx]])

    return generated_text


# --- Execution ---

# Option A: Define a manual start string (ensure it matches your file format exactly)
# If your files start with "FORMAT_X", use that.
# manual_start = "SPEC_TYPE_A"

# Option B (Recommended): Automatically grab the start from your training data
# This guarantees no KeyErrors.
auto_start = training_file_content.strip()[:10]

print(f"\n--- Generating starting with: '{auto_start}' ---\n")
generated_result = generate_format_spec(model, tokenizer, auto_start, predict_len=300)
print(generated_result)