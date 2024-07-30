import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from linformer import Linformer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# 定义自定义数据集类
class RandomDataset(Dataset):
    def __init__(self, seq_length, num_samples):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.data = torch.rand((num_samples, seq_length, 1))  # 随机生成数据

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# 定义编码器类
class LinformerSeqEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, seq_length=10000, num_heads=4, num_layers=1):
        super(LinformerSeqEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=seq_length)

        self.linformer = Linformer(
            dim=embed_dim,
            seq_len=seq_length,
            depth=num_layers,
            heads=num_heads,
            k=256  # 投影长度，可以根据需要调整
        )

        self.fc = nn.Linear(embed_dim, embed_dim // 2)

    def forward(self, x):

        # (batch, seq_len, emb_dim)
        x = x.to(device)

        # Embedding
        x = self.embedding(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Linformer encoding
        x = self.linformer(x)

        # Dimensionality reduction
        x = self.fc(x)
        return x


# 定义解码器类
class LinformerSeqDecoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(LinformerSeqDecoder, self).__init__()
        self.fc = nn.Linear(embed_dim // 2, embed_dim)

        self.linformer = Linformer(
            dim=embed_dim,
            seq_len=10000,
            depth=1,
            heads=4,
            k=256
        )

        self.positional_encoding = PositionalEncoding(embed_dim, max_len=10000)
        self.embedding = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        x = x.to(device)

        # Dimensionality expansion
        x = self.fc(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Linformer decoding
        x = self.linformer(x)

        # Embedding
        x = self.embedding(x)
        return x


# 定义自编码器类
class LinformerSeqAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(LinformerSeqAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 数据生成
batch_size = 10
seq_length = 10000
num_samples = 300

dataset = RandomDataset(seq_length, num_samples)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建编码器和解码器实例
input_dim = 1
embed_dim = 200
encoder = LinformerSeqEncoder(input_dim=input_dim, embed_dim=embed_dim, seq_length=seq_length).to(device)
decoder = LinformerSeqDecoder(input_dim=input_dim, embed_dim=embed_dim).to(device)

# 创建自编码器实例
autoencoder = LinformerSeqAutoencoder(encoder, decoder).to(device)

# 打印模型结构
print("Encoder Structure:")
print(encoder)

print("\nDecoder Structure:")
print(decoder)

# 打印模型参数量统计
print("\nAutoencoder Summary:")
summary(autoencoder, input_size=(seq_length, input_dim))

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练自编码器
max_epochs = 100
loss_threshold = 30  # 损失30轮不下降就早停

min_loss = np.inf
no_improvement_count = 0

# 记录每个epoch的损失
train_losses = []

for epoch in range(max_epochs):
    total_loss = 0.0
    # 使用tqdm包装train_loader以显示进度条
    for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", leave=False):
        inputs = data.float().to(device)

        # 前向传播
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 记录每个epoch的平均损失
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}")

    # 早停策略
    if avg_loss < min_loss:
        min_loss = avg_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= loss_threshold:
        print(f"No improvement for {loss_threshold} epochs, training stopped.")
        break

print("Training finished.")

# 保存模型
torch.save(autoencoder.state_dict(), 'linformer_autoencoder.pth')
print("Model saved.")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
