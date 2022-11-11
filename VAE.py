import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(8000, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, inp):

        enc = self.enc(inp)
        return enc


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec = nn.Sequential(
            nn.Linear(256, 8000),
            nn.Sigmoid(),
        )

    def forward(self, inp ):
        ep = torch.randn(8,1,128).cuda()
        mean = inp[:, :, :128].cuda()
        stddev = inp[:, :, 128:].cuda()
        inp = mean+ep*stddev.cuda()
        dec = self.dec(inp)


        return dec,mean,stddev

#USE Decoder(Encoder(input))

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self,input):
        tensor = self.enc(input)

        out,mean, stddev = self.dec(tensor)
        #print("out:",out.shape)
        #print("mean:",mean.shape)
        #print("stddev:",stddev.shape)
        return out, mean, stddev


class VAE_V2(nn.Module):
    def __init__(self, x_dim, h_dim1, z_dim):
        super(VAE_V2, self).__init__()
        self.hidden_dim = h_dim1
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc21 = nn.Linear(h_dim1, z_dim)
        self.fc22 = nn.Linear(h_dim1, z_dim)
        # decoder part
        self.fc3 = nn.Linear(z_dim, h_dim1)
        self.fc4 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        x = self.fc1(x)
        h = F.relu(x)
        return self.fc21(h), self.fc22(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

class VAEV3(nn.Module):
    def __init__(self):
        super(VAEV3, self).__init__()

        self.fc1 = nn.Linear(8000, 256)
        self.fc21 = nn.Linear(256, 20)
        self.fc22 = nn.Linear(256, 20)
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 8000)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 8000))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar