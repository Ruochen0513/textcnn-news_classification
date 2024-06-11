<<<<<<< HEAD
import jieba
import csv
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 读取文本文件
with open('cnews.train.txt', 'r', encoding='utf-8') as in_file:
    lines = in_file.readlines()

# 处理每一行并写入CSV文件
with open('cnews.train.csv', 'w', newline='', encoding='utf-8') as out_file:
    writer = csv.writer(out_file)
    for line in lines:
        # 提取每一行的前两个字符和剩余部分
        first_two = line[:2]
        rest = line[3:].strip()
        # 写入CSV文件
        writer.writerow([first_two, rest])

# 数据存储在CSV文件中，有两列："category" 和 "content"
df = pd.read_csv('cnews.train.csv')

columns = ['category', 'content']
df.columns = columns


# 文本清洗函数
def clean_text(text):
    # 去除HTML标签
    text = text.replace('<[^>]*>', '')
    # 去除特殊字符和标点符号
    text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in ".,?!;:\"'")
    # 转换为小写
    text = text.lower()
    return text


# 清洗文本
df['content'] = df['content'].apply(clean_text)


# 分词
def tokenize(text):
    return list(jieba.cut(text))


df['tokens'] = df['content'].apply(tokenize)


# 构建词汇表
def build_vocab(tokens):
    word_counts = Counter(tokens)
    # 设置词汇表大小
    vocab_size = 50000
    # 过滤低频词，保留高频词
    vocab = [word for word, count in word_counts.most_common(vocab_size)]
    # 添加一个特殊标记用于填充（padding）
    vocab.append('<PAD>')
    # 添加一个特殊标记用于未知词（unknown words）
    vocab.append('<UNK>')
    return vocab, {word: idx for idx, word in enumerate(vocab)}


# 合并所有token以构建全局词汇表
all_tokens = [token for news_tokens in df['tokens'] for token in news_tokens]
vocab, word_to_idx = build_vocab(all_tokens)


# 文本表示：将tokens转换为索引列表，并填充到相同长度
def tokens_to_indices(tokens, word_to_idx, max_seq_length):
    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    # 填充或截断到max_seq_length
    indices += [word_to_idx['<PAD>']] * (max_seq_length - len(indices))
    indices = indices[:max_seq_length]
    return np.array(indices)


# 设定最大序列长度
max_seq_length = 200
df['indices'] = df['tokens'].apply(lambda tokens: tokens_to_indices(tokens, word_to_idx, max_seq_length))

# category列是文本类别，转换为数值编码
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# 将 indices 列转换为字符串以便后续正确解析
df['indices'] = df['indices'].apply(lambda x: ','.join(map(str, x)))
# 划分数据集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # 80%训练集，20%测试集
train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=42)  # 再将训练集划分为70%训练集和30%验证集

np.save('classes.npy', label_encoder.classes_)  # 保存类别编码
np.save('word_to_idx.npy', word_to_idx)  # 保存词汇表

# 保存预处理后的数据
train_df.to_csv('train_preprocessed.csv', index=False)
val_df.to_csv('val_preprocessed.csv', index=False)
test_df.to_csv('test_preprocessed.csv', index=False)
=======
import jieba
import csv
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 读取文本文件
with open('cnews.train.txt', 'r', encoding='utf-8') as in_file:
    lines = in_file.readlines()

# 处理每一行并写入CSV文件
with open('cnews.train.csv', 'w', newline='', encoding='utf-8') as out_file:
    writer = csv.writer(out_file)
    for line in lines:
        # 提取每一行的前两个字符和剩余部分
        first_two = line[:2]
        rest = line[3:].strip()
        # 写入CSV文件
        writer.writerow([first_two, rest])

# 数据存储在CSV文件中，有两列："category" 和 "content"
df = pd.read_csv('cnews.train.csv')

columns = ['category', 'content']
df.columns = columns


# 文本清洗函数
def clean_text(text):
    # 去除HTML标签
    text = text.replace('<[^>]*>', '')
    # 去除特殊字符和标点符号
    text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in ".,?!;:\"'")
    # 转换为小写
    text = text.lower()
    return text


# 清洗文本
df['content'] = df['content'].apply(clean_text)


# 分词
def tokenize(text):
    return list(jieba.cut(text))


df['tokens'] = df['content'].apply(tokenize)


# 构建词汇表
def build_vocab(tokens):
    word_counts = Counter(tokens)
    # 设置词汇表大小
    vocab_size = 50000
    # 过滤低频词，保留高频词
    vocab = [word for word, count in word_counts.most_common(vocab_size)]
    # 添加一个特殊标记用于填充（padding）
    vocab.append('<PAD>')
    # 添加一个特殊标记用于未知词（unknown words）
    vocab.append('<UNK>')
    return vocab, {word: idx for idx, word in enumerate(vocab)}


# 合并所有token以构建全局词汇表
all_tokens = [token for news_tokens in df['tokens'] for token in news_tokens]
vocab, word_to_idx = build_vocab(all_tokens)


# 文本表示：将tokens转换为索引列表，并填充到相同长度
def tokens_to_indices(tokens, word_to_idx, max_seq_length):
    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    # 填充或截断到max_seq_length
    indices += [word_to_idx['<PAD>']] * (max_seq_length - len(indices))
    indices = indices[:max_seq_length]
    return np.array(indices)


# 设定最大序列长度
max_seq_length = 200
df['indices'] = df['tokens'].apply(lambda tokens: tokens_to_indices(tokens, word_to_idx, max_seq_length))

# category列是文本类别，转换为数值编码
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# 将 indices 列转换为字符串以便后续正确解析
df['indices'] = df['indices'].apply(lambda x: ','.join(map(str, x)))
# 划分数据集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # 80%训练集，20%测试集
train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=42)  # 再将训练集划分为70%训练集和30%验证集

np.save('classes.npy', label_encoder.classes_)  # 保存类别编码
np.save('word_to_idx.npy', word_to_idx)  # 保存词汇表

# 保存预处理后的数据
train_df.to_csv('train_preprocessed.csv', index=False)
val_df.to_csv('val_preprocessed.csv', index=False)
test_df.to_csv('test_preprocessed.csv', index=False)
>>>>>>> e96aa8490b06947fea5d0ce7d6c33354a735b0fa
