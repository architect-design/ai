# services.py
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from tokenizer import FormatTokenizer
from model import SpecificationLearner

# Global storage
model_registry = {}
MODELS_DIR = "saved_models"


class ModelTrainer:

    @staticmethod
    def save_model(format_name, model, tokenizer):
        """Saves model weights and tokenizer to disk."""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        folder_path = os.path.join(MODELS_DIR, format_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save Model Weights
        torch.save(model.state_dict(), os.path.join(folder_path, "model.pt"))

        # Save Tokenizer
        tokenizer.save(os.path.join(folder_path, "tokenizer.json"))

        # Save Model Config (architecture details)
        config = {
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": 64,  # Must match the value in train_format
            "hidden_dim": 128  # Must match the value in train_format
        }
        with open(os.path.join(folder_path, "config.json"), 'w') as f:
            json.dump(config, f)

        print(f"Model '{format_name}' saved to disk.")

    @staticmethod
    def load_model(format_name):
        """Loads a specific model from disk into memory."""
        folder_path = os.path.join(MODELS_DIR, format_name)
        config_path = os.path.join(folder_path, "config.json")
        model_path = os.path.join(folder_path, "model.pt")
        tokenizer_path = os.path.join(folder_path, "tokenizer.json")

        if os.path.exists(config_path) and os.path.exists(model_path):
            # Load Config
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load Tokenizer
            tokenizer = FormatTokenizer()
            tokenizer.load(tokenizer_path)

            # Initialize Model
            model = SpecificationLearner(
                config['vocab_size'],
                config['embedding_dim'],
                config['hidden_dim']
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set to evaluation mode

            # Register
            model_registry[format_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            print(f"Loaded existing model: {format_name}")
            return True
        return False

    @staticmethod
    def load_all_models():
        """Scans the saved_models directory and loads all existing models."""
        if not os.path.exists(MODELS_DIR):
            return

        for format_name in os.listdir(MODELS_DIR):
            if os.path.isdir(os.path.join(MODELS_DIR, format_name)):
                ModelTrainer.load_model(format_name)

    @staticmethod
    def train_format(format_name, raw_text, epochs=50):
        # Check if already loaded
        if format_name in model_registry:
            print(f"Model '{format_name}' already exists. Skipping training.")
            return

        print(f"Training new model for: {format_name}")

        tokenizer = FormatTokenizer()
        tokenizer.fit(raw_text)

        vocab_size = tokenizer.vocab_size
        encoded = tokenizer.encode(raw_text)

        seq_length = 20
        inputs, targets = [], []
        for i in range(0, len(encoded) - seq_length):
            chunk = encoded[i:i + seq_length + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])

        if len(inputs) == 0:
            raise ValueError("File content is too short to train.")

        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        embedding_dim = 64
        hidden_dim = 128
        model = SpecificationLearner(vocab_size, embedding_dim, hidden_dim)

        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()

        batch_size = 64
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch_inputs, batch_targets in loader:
                current_batch_size = batch_inputs.size(0)
                hidden = model.init_hidden(current_batch_size)

                optimizer.zero_grad()
                output, hidden = model(batch_inputs, hidden)
                loss = criterion(output, batch_targets.view(-1))
                loss.backward()
                optimizer.step()

        # Save to disk after training
        ModelTrainer.save_model(format_name, model, tokenizer)

        # Register in memory
        model_registry[format_name] = {
            "model": model,
            "tokenizer": tokenizer
        }

        # services.py (Only the generate_text method needs changing)

        # services.py

    @staticmethod
    def generate_text(format_name, start_str="", predict_len=200, mode="chars"):
        if format_name not in model_registry:
            return "Error: Format not trained yet."

        data = model_registry[format_name]
        model = data["model"]
        tokenizer = data["tokenizer"]

        model.eval()

        # Handle start string
        if not start_str:
            start_str = tokenizer.get_start_char()
        else:
            valid_start = True
            for char in start_str:
                if char not in tokenizer.char_to_int:
                    valid_start = False
                    break
            if not valid_start:
                start_str = tokenizer.get_start_char()

        hidden = model.init_hidden(1)

        try:
            input_seq = torch.tensor([tokenizer.encode(start_str)])
        except KeyError:
            return "Error: Start string contains invalid characters."

        generated_text = start_str

        # --- FIXED LOGIC ---
        line_count = generated_text.count('\n')
        char_count = len(generated_text)

        # Safety limit: Stop if we generate 5x the expected characters
        # (prevents infinite loop if model forgets to hit 'Enter')
        # Or a hard limit of 15,000 characters for performance.
        if mode == "lines":
            max_chars = min(predict_len * 150, 15000)
        else:
            max_chars = predict_len

        with torch.no_grad():
            while True:
                # 1. Check Stop Conditions
                if mode == "lines":
                    if line_count >= predict_len:
                        break
                else:  # mode == "chars"
                    if char_count >= predict_len:
                        break

                # Safety break (prevent browser hang)
                if char_count >= max_chars:
                    print(f"Safety limit reached ({max_chars} chars). Stopping generation.")
                    break

                # 2. Generate Next Char
                output, hidden = model(input_seq, hidden)
                last_char_logits = output[-1, :].unsqueeze(0)
                probs = torch.softmax(last_char_logits, dim=1)
                top_idx = torch.multinomial(probs, 1).item()

                char = tokenizer.decode([top_idx])
                generated_text += char

                # 3. Update Counts
                if char == '\n':
                    line_count += 1
                char_count += 1

                # 4. Prepare next input
                input_seq = torch.tensor([[top_idx]])

        return generated_text