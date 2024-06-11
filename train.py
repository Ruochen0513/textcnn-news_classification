<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import TextCNN

# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, df):
        self.labels = df['category_encoded'].values
        # 将indices列转换为数值类型数组
        self.texts = df['indices'].apply(lambda x: np.array(list(map(int, x.replace('\n', '').split(','))))).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# 训练函数
def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    model.to(device)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))

        # 验证集上评估
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

# 验证函数
def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)


if __name__ == "__main__":
    # 加载预处理后的数据
    train_df = pd.read_csv('train_preprocessed.csv')
    val_df = pd.read_csv('val_preprocessed.csv')
    test_df = pd.read_csv('test_preprocessed.csv')

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    num_classes = len(train_df['category_encoded'].unique())
    vocab_size = 50002  # 50000 + <PAD> + <UNK>
    embedding_dim = 128
    kernel_sizes = [3, 4, 5]
    num_channels = [100, 100, 100]
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    max_seq_length = 200

    # 创建数据加载器
    train_dataset = TextDataset(train_df)
    val_dataset = TextDataset(val_df)
    test_dataset = TextDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型、损失函数和优化器
    model = TextCNN(num_classes=num_classes,
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    kernel_sizes=kernel_sizes,
                    num_channels=num_channels)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train(model, criterion, optimizer, train_loader, val_loader, num_epochs, device)
=======
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import TextCNN

# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, df):
        self.labels = df['category_encoded'].values
        # 将indices列转换为数值类型数组
        self.texts = df['indices'].apply(lambda x: np.array(list(map(int, x.replace('\n', '').split(','))))).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# 训练函数
def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    model.to(device)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))

        # 验证集上评估
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

# 验证函数
def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)


if __name__ == "__main__":
    # 加载预处理后的数据
    train_df = pd.read_csv('train_preprocessed.csv')
    val_df = pd.read_csv('val_preprocessed.csv')
    test_df = pd.read_csv('test_preprocessed.csv')

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    num_classes = len(train_df['category_encoded'].unique())
    vocab_size = 50002  # 50000 + <PAD> + <UNK>
    embedding_dim = 128
    kernel_sizes = [3, 4, 5]
    num_channels = [100, 100, 100]
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    max_seq_length = 200

    # 创建数据加载器
    train_dataset = TextDataset(train_df)
    val_dataset = TextDataset(val_df)
    test_dataset = TextDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型、损失函数和优化器
    model = TextCNN(num_classes=num_classes,
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    kernel_sizes=kernel_sizes,
                    num_channels=num_channels)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train(model, criterion, optimizer, train_loader, val_loader, num_epochs, device)
>>>>>>> e96aa8490b06947fea5d0ce7d6c33354a735b0fa
