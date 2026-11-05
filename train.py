import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 78, 128)
        self.fc2 = nn.Linear(128, 2) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # 【新增】防止模型死记硬背

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # 【新增】
        x = self.fc2(x)
        return x

def main():
    print("加载数据中...")
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    all_samples = data[0] + data[1] + data[2]
    
    X, y = [], []
    for (mfcc, device), label in all_samples:
        X.append(mfcc)
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    
    print("正在对数据进行标准化...")
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    X = torch.tensor(X).unsqueeze(1) 
    y = torch.tensor(np.array(y), dtype=torch.long)
    
    # 【新增】划分训练集和测试集（8:2），看看模型是不是真的学到了东西
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # 【修改】学习率调小为 0.0001
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
    epochs = 10
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 测试阶段（看看真实指标）
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, 测试集准确率: {acc:.2f}%")
    
    torch.save(model.state_dict(), 'model_weights.pkl')
    print("训练完成！权重已保存为 model_weights.pkl，请把这个文件交上去！")

if __name__ == '__main__':
    main()