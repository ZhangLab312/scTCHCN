import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义闵可夫斯基损失函数
class MinkowskiLoss(nn.Module):
    def __init__(self, p=3):
        super(MinkowskiLoss, self).__init__()
        self.p = p

    def forward(self, x, x_hat):
        # 对每个样本计算闵可夫斯基距离
        loss_per_sample = torch.sum(torch.abs(x - x_hat) ** self.p, dim=1) ** (1 / self.p)
        # 对所有样本的损失取平均值
        return torch.mean(loss_per_sample)


# 定义简单的自动编码器模型
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 生成示例数据（调整为更大的范围）
torch.manual_seed(0)
data = torch.rand(1000, 50)  # 1000个样本，每个样本50维

# 在数据上添加噪声
noisy_data = data + 0.5 * torch.randn_like(data)

# 将数据移动到GPU
data = data.to(device)
noisy_data = noisy_data.to(device)

# 定义模型，并移动到GPU
model_minkowski = SimpleAutoencoder(input_dim=50, hidden_dim=25).to(device)

# 定义损失函数和优化器
criterion_minkowski = MinkowskiLoss(p=50)
optimizer_minkowski = optim.Adam(model_minkowski.parameters(), lr=0.0025)

# 训练模型
num_epochs = 5000
losses_minkowski = []

for epoch in range(num_epochs):
    optimizer_minkowski.zero_grad()
    outputs_minkowski = model_minkowski(noisy_data)
    loss_minkowski = criterion_minkowski(data, outputs_minkowski)
    loss_minkowski.backward()
    optimizer_minkowski.step()
    losses_minkowski.append(loss_minkowski.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Minkowski Loss: {loss_minkowski.item():.4f}')

# 生成损失下降曲线图
plt.figure(figsize=(10, 6))
plt.plot(losses_minkowski, label='Minkowski Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Minkowski Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
