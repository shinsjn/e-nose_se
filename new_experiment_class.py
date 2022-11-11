import matplotlib.pyplot as plt
from cnn1d import *
#from model.default_model import *

from utility.data_loader.RawData_class import RawData
from torch.utils.data import Dataset,DataLoader,random_split
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler

import VAE
from VAE import VAE_V2
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
def loss_function(recon_x, x, mu, log_var):
    BCE = F.cross_entropy(recon_x, x.view(-1,8000), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #print("BCE: ",BCE)
    #print("KLD: ", KLD)
    #print("mu, log_var: ", mu, log_var)
    return BCE + KLD

class Custum_Dataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        # 필요한 변수 선언

        """
        opt_data: 'train', 'validation'
        """
        temp = np.loadtxt("C:/Users/shins/Desktop/MLPA/E-nose/code/shin_prof_code/enose_codes/codes/concatdata.dat")
        self.x = temp[:, :-1]
        #print(self.x.shape)
        self.x = StandardScaler().fit_transform(self.x)
        self.y = temp[:, -1]

        # ===

    def __getitem__(self, index):
        # index번째 data를 return하도록 코드를 짜야한다.
        xx = torch.FloatTensor(self.x[index])
        yy = torch.tensor(self.y[index])

        return xx, yy

    def __len__(self):
        return len(self.y)

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

class Experiment():
    def __init__(self, arg):
        self.dae_epochs = arg['dae_epochs']
        self.classifier_epochs = arg['classifier_epochs']
        self.noise = arg['noise_ratio']
        self.train_subsampler = None
        self.test_subsampler = None
        self.batch_size = arg['batch_size']
        self.dae_lr = arg['dae_lr']
        self.clf_lr = arg['clf_lr']
    def data_init(self):
        self.rd = RawData(self.sensor_type)
        self.rd.load_data()
        self.rd.x_y_split(0.8, 0, 0.2)
        self.rd.resize_data()



    def one_hot_encoding(self, inp):
        num = np.unique(inp, axis=0)
        num = num.shape[0]
        enc = np.eye(num)[inp]
        return enc.astype('float32')

    def train_init(self):
        # gpu available?
        print("GPU available:", torch.cuda.is_available(), " GPU_name:", torch.cuda.get_device_name())
        self.device = torch.device('cuda')

        #dataset = Custum_Dataset()
        #======test dataset define======
        temp = np.loadtxt("C:/Users/shins/Desktop/MLPA/E-nose/code/shin_prof_code/enose_codes/codes/concatdata.dat")

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
        #===========


        train_dataset,val_dataset,test_dataset = random_split(dataset,[int(len(dataset)*0.8),0,int(len(dataset)*0.2)])



        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,shuffle=True)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size)

        # model_init
        self.model = VAE_V2(x_dim=8000,h_dim1=256,z_dim=200)
        self.model = self.model.cuda()
        self.model.apply(reset_weights)

        self.classifier = CNN_1d(channel1=50, channel2=100, lin1=250, out_size=7, ker_size=9, pool_size=2)
        # set model to train mode
        self.classifier = self.classifier.cuda()
        self.model.train()
        self.classifier.train()

    def train_dae(self):
        print("Train Denoising AutoEncdoer...")
        torch.autograd.set_detect_anomaly(True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.dae_lr)
        criterion = nn.MSELoss()
        loss_list=[]
        for epoch in range(self.dae_epochs):
            for batch_ind, samples in enumerate(self.train_loader):
                x_t, _ = samples
                x_t = x_t.reshape([-1,1,8000])
                y_t = x_t
                # x_t에 noise 추가
                noise_gate = self.noise  # 추가해줄 노이즈의 정도
                noise = torch.randn(x_t.shape)*noise_gate
                #x_t = x_t + noise

                x_t, y_t = x_t.to(self.device), y_t.to(self.device)


                # predict
                pred,mean,stddev = self.model(x_t.float())
                pred = pred.view([-1, 8000])

                # calculate loss
                #recon_loss = criterion(pred, y_t.float())
                #epsilon = 1e-8
                #vae_loss = torch.sum(0.5 * (torch.square(mean) + torch.square(stddev) - 2.0 * torch.log(stddev + epsilon) - 1.0))

                #loss = vae_loss + recon_loss
                # loss = vae_loss + recon_loss
                loss = loss_function(pred,y_t,mean,stddev)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch%10==0:
                print("epochs: ", str(epoch), "/", str(self.dae_epochs))
                print(loss.item())
                loss_list.append(loss.item())

        plt.title('DAE_LOSS')
        print(list(range(0,len(loss_list))))
        plt.plot(loss_list)
        plt.show()

    def train_classifier(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from sklearn.model_selection import StratifiedKFold
        from sklearn import preprocessing
        from cnn1d import CNN_1d
        from tqdm import trange

        best_acc = -1
        # temp = np.loadtxt('./data/NEW_e_nose_data_2022.dat')
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

        # torch.save(dataset, 'tensor.pt')

        # dataset = torch.load('tensor.pt', map_location=lambda storage, loc: storage)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        k_folds = 4
        batch_size = self.batch_size
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
        learning_rate = self.clf_lr
        epochs = 300
        criterion = torch.nn.CrossEntropyLoss()
        results = {}
        label_sum =[0,0,0,0,0,0,0]
        predict_sum = [0, 0, 0, 0, 0, 0, 0]
        import time
        start = time.time()
        for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)):
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset.__len__(), sampler=test_subsampler)

            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
            loss_list = []
            tepoch = trange(epochs, desc="Epochs")
            for epoch in tepoch:
                tmploss_list = []
                for i, (inputs, labels) in enumerate(trainloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    out,mu,log_var = self.model(inputs)
                    y_pred = self.classifier(out)
                    loss = criterion(y_pred, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_list.append(loss.item())
                if epoch % 20 == 0:
                    # print("Loss:", loss.item())
                    tepoch.set_postfix(loss=loss.item())
            # Evaluationfor this fold
            correct, total = 0, 0
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    out, mu, log_var = self.model(inputs)
                    outputs = self.classifier(out)

                    for value in targets:
                        label_sum[value] = label_sum[value] + 1

                    _, predicted = torch.max(outputs, 1)
                    for value in predicted:
                        predict_sum[value] = predict_sum[value] + 1
                        print("predicted: ",predicted)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    print('\nActual   :', targets)
                    print('Predicted:', predicted)
                    print('total:', total, ' Correct:', correct)
                # Print accuracy
                print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
                print('--------------------------------')
                results[fold] = 100.0 * (correct / total)
            step = np.linspace(0, epochs, epochs)
            plt.plot(step, np.array(loss_list))
            plt.xlabel('epoch')
            plt.ylabel('loss')
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum / len(results.items())} %')
        if best_acc < sum / len(results.items()):
            best_acc = sum / len(results.items())
            best_lr = self.clf_lr
        step = np.linspace(0, epochs, epochs)
        end = time.time()
        elapsed = round(end - start)
        # print('Time:', end - start)
        # plt.plot(step, np.array(loss_list))
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        print("label sum: ",label_sum)
        print("predict sum: ", predict_sum)
        return best_acc

    def test_acc(self):
        self.model.eval()
        self.classifier.eval()
        test_accuracy = 0.0

        for param in self.model.parameters():
            param.requires_grad=False

        optimizer = optim.Adam(self.classifier.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            # test accuracy 계산
            for batch_ind, samples in enumerate(self.test_loader):
                x_t, y_t = samples
                x_t, y_t = x_t.to(self.device), y_t.to(self.device)
                x_t = x_t.reshape([-1,1,8000])

                x_t, m, s = self.model(x_t.float())
                # predict
                pred = self.classifier(x_t)
                y_t = torch.flatten(y_t)
                pred = torch.argmax(pred,dim=1)
                y_t = y_t.float()

                # calculate loss
                loss = criterion(pred, y_t)

                correct = 0

                correct += (pred == y_t).sum().item()
                test_accuracy = 100 * (correct / y_t.size(0))
                print("Classifier's train_accuracy: ", test_accuracy)
        self.model.train()
        self.classifier.train()

    def test_acc_noise_added(self, noise_gate):
        self.noise_test_x = list()
        # test_x에 noise 추가
        for xx in self.test_x:
            uniform_dist = np.random.rand(xx.shape[0])
            noise = np.where(uniform_dist > noise_gate, 1, 0)
            self.noise_test_x.append(noise * xx)
        self.noise_test_dataset = Dataset(self.noise_test_x, self.test_y)
        self.noise_test_loader = data.DataLoader(dataset=self.noise_test_dataset, batch_size=256, shuffle=True)
        self.test_acc(self.noise_test_loader)

