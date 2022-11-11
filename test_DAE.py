import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
from sklearn import preprocessing
from torch.utils.data import Dataset,DataLoader,random_split


import matplotlib.pyplot as plt
import numpy as np

# 하이퍼파라미터
EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)

temp = np.loadtxt('C:/Users/shins/Desktop/MLPA/E-nose/code/shin_prof_code/enose_codes/codes/concatdata.dat')

x = temp[:, :-1]
y = temp[:, -1]

#print(x.shape)
#print(y.shape)

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
x = x.reshape(x.shape[0], 1, x.shape[1])
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y).type(torch.LongTensor)
dataset = torch.utils.data.TensorDataset(x, y)
train_dataset, val_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.8), 0, int(len(dataset) * 0.2)])

train_loader = DataLoader(
            train_dataset,
            batch_size=32,shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(8000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),  # 입력의 특징을 3차원으로 압축합니다
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 8000),
            nn.Sigmoid(),  # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0
    for step, (x, label) in enumerate(train_loader):
        noisy_x = add_noise(x)  # 입력에 노이즈 더하기
        noisy_x = noisy_x.view(-1, 8000).to(DEVICE)
        y = x.view(-1, 8000).to(DEVICE)

        label = label.to(DEVICE)
        encoded, decoded = autoencoder(noisy_x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    return avg_loss / len(train_loader)


for epoch in range(1, EPOCH + 1):
    loss = train(autoencoder, train_loader)
    print("[Epoch {}] loss:{}".format(epoch, loss))
    # 이번 예제에선 학습시 시각화를 건너 뜁니다

# # 이미지 복원 시각화 하기

# 모델이 학습시 본적이 없는 데이터로 검증하기 위해 테스트 데이터셋을 가져옵니다.


# 테스트셋에서 이미지 한장을 가져옵니다.
sample_data = test_dataset[0][0][0].view(-1, 8000)

# 이미지를 add_noise로 오염시킨 후, 모델에 통과시킵니다.
original_x = sample_data[0]
noisy_x = add_noise(original_x).to(DEVICE)
_, recovered_x = autoencoder(noisy_x)


# 시각화를 위해 넘파이 행렬로 바꿔줍니다.
original_img = np.reshape(original_x.to("cpu").data.numpy(), (8000))
noisy_img = np.reshape(noisy_x.to("cpu").data.numpy(), (8000))
recovered_img = np.reshape(recovered_x.to("cpu").data.numpy(), (8000))
print('origin: ', original_img)
print('noisy: ', noisy_img)
print('recover: ', recovered_img)

plt.scatter(np.array(range(8000)),original_img)
plt.title('origin')
plt.show()

plt.scatter(np.array(range(8000)),noisy_img)
plt.title('noisy')
plt.show()

plt.scatter(np.array(range(8000)),recovered_img)
plt.title('recover')
plt.show()