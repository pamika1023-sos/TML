# DCNN Image Classification on CIFAR-10 Dataset using PyTorch

โครงงานนี้เป็นส่วนหนึ่งของใบงาน (LAB: DCNN on a Dataset of Your Choice)[cite: 2] เกี่ยวกับการประยุกต์ใช้งาน **Deep Convolutional Neural Network (DCNN)** สำหรับจำแนกรูปภาพจากชุดข้อมูล **CIFAR-10** ด้วย PyTorch[cite: 1]

---

## 🎯 วัตถุประสงค์ (Objective)
* ประยุกต์ใช้โมเดล DCNN ในการจำแนกรูปภาพจากชุดข้อมูลที่เลือก (CIFAR-10)[cite: 2]
* เปรียบเทียบประสิทธิภาพการเรียนรู้ของโมเดลภายใต้จำนวนรอบ (Epochs) และโครงสร้าง DCNN ที่กำหนด[cite: 2]

---

## 📋 รายละเอียดการดำเนินงาน (Contents)
1. **Dataset Selection & Preparation**: เลือกใช้ชุดข้อมูลมาตรฐาน `CIFAR-10` ซึ่งประกอบด้วยภาพสีขนาด $32 \times 32$ พิกเซล จำนวน 10 คลาส
2. **Data Splitting & Standardization**: แบ่งชุดข้อมูลเป็น Training Set และ Testing Set พร้อมทำ Standardize ข้อมูลให้อยู่ในช่วงมาตรฐานด้วย `transforms.Normalize`[cite: 1, 2]
3. **Model Architecture**: สร้างโครงสร้าง DCNN แบบ 2 ชั้น Conv2d ร่วมกับ MaxPool2d และ Fully Connected Layers[cite: 1, 2]
4. **Model Training**: ฝึกสอนโมเดลด้วยจำนวน 5 Epochs โดยใช้ Optimizer แบบ Adam และ Loss Function แบบ CrossEntropyLoss[cite: 1, 2]
5. **Evaluation**: ประเมินผลความแม่นยำ (Accuracy) ของโมเดลบนชุดข้อมูลทดสอบ (Test Set)[cite: 1, 2]

---

## 🛠️ โค้ดการดำเนินงาน (Source Code)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

print("--- 1. โหลดและเตรียมชุดข้อมูล (CIFAR-10) ด้วย PyTorch ---")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("--- 2. สร้างโครงสร้างโมเดล DCNN Configuration ---")
class SimpleDCNN(nn.Module):
    def __init__(self):
        super(SimpleDCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleDCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("--- 3. เริ่มกระบวนการ Train โมเดล (Epochs) ---")
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(trainloader):.4f}")

print("--- 4. ประเมินผลความแม่นยำ (Accuracy) บนชุดข้อมูล Test ---")
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n================ ผลการทดสอบ ================")
print(f"Accuracy ของโมเดลบน Test Set: {accuracy:.2f}%")
print(f"============================================")