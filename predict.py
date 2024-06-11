<<<<<<< HEAD
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import jieba
from model import TextCNN
from train import TextDataset
from sklearn.preprocessing import LabelEncoder


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def predict_text(model, text, word_to_idx, max_seq_length, device):
    tokens = list(jieba.cut(text))
    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    indices += [word_to_idx['<PAD>']] * (max_seq_length - len(indices))
    indices = indices[:max_seq_length]
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


if __name__ == "__main__":
    test_df = pd.read_csv('test_preprocessed.csv')

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    word_to_idx = np.load('word_to_idx.npy', allow_pickle=True).item()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(label_encoder.classes_)
    vocab_size = 50002
    embedding_dim = 128
    kernel_sizes = [3, 4, 5]
    num_channels = [100, 100, 100]
    max_seq_length = 200

    test_dataset = TextDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = TextCNN(num_classes=num_classes,
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    kernel_sizes=kernel_sizes,
                    num_channels=num_channels)

    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)

    print("Evaluating on test data...")
    evaluate(model, test_loader, device)

    while True:
        text = input("Enter a piece of text (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        predicted_class = predict_text(model, text, word_to_idx, max_seq_length, device)
        print(f"Predicted class: {label_encoder.inverse_transform([predicted_class])[0]}")
=======
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import jieba
from model import TextCNN
from train import TextDataset
from sklearn.preprocessing import LabelEncoder


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def predict_text(model, text, word_to_idx, max_seq_length, device):
    tokens = list(jieba.cut(text))
    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    indices += [word_to_idx['<PAD>']] * (max_seq_length - len(indices))
    indices = indices[:max_seq_length]
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


if __name__ == "__main__":
    test_df = pd.read_csv('test_preprocessed.csv')

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    word_to_idx = np.load('word_to_idx.npy', allow_pickle=True).item()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(label_encoder.classes_)
    vocab_size = 50002
    embedding_dim = 128
    kernel_sizes = [3, 4, 5]
    num_channels = [100, 100, 100]
    max_seq_length = 200

    test_dataset = TextDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = TextCNN(num_classes=num_classes,
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    kernel_sizes=kernel_sizes,
                    num_channels=num_channels)

    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)

    print("Evaluating on test data...")
    evaluate(model, test_loader, device)

    while True:
        text = input("Enter a piece of text (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        predicted_class = predict_text(model, text, word_to_idx, max_seq_length, device)
        print(f"Predicted class: {label_encoder.inverse_transform([predicted_class])[0]}")
>>>>>>> e96aa8490b06947fea5d0ce7d6c33354a735b0fa
