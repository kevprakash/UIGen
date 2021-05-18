import torch
from torch import nn
import math

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class UIGen(torch.nn.Module):
    def __init__(self, I, L, d, activation=None):
        super().__init__()

        # Save hyperparamters
        self.I = I
        self.L = L
        self.d = d
        self.activation = activation

        # Constants
        div = torch.Tensor([(i + 1.0) for i in range(L)])
        div = torch.unsqueeze(div, 0)
        div = div.repeat(L, 1)
        self.PA = torch.tril(torch.ones((L, L)))/div
        self.SA = torch.tril(torch.ones((L, L)))
        self.zero = torch.zeros((1, d))

        # Learned parameters
        self.DE = nn.Parameter(torch.Tensor(self.I, self.d))                    # Dimension embedding weights
        self.PE = nn.Parameter(torch.Tensor(self.L, self.d))                    # Position embedding weights
        self.D1 = nn.Parameter(torch.Tensor(self.d * 2, self.d))                # First dense layer weights
        self.D2 = nn.Parameter(torch.Tensor(self.d, self.I))                    # Second dense layer weights
        self.Att = nn.Parameter(torch.Tensor(self.d * 2, self.d))               # Attention weights

        # Parameter Initialization
        print("Initializing model with hyperparameters:" +
              "\nI=" + str(self.I) +
              "\nL=" + str(self.L) +
              "\nd=" + str(self.d))
        for p in self.parameters():
            nn.init.kaiming_normal_(p, a=math.sqrt(5))

    def forward(self, x):
        # Embedding
        dimEmbed = torch.mm(x, self.DE)
        posEmbed = dimEmbed + self.PE                                           # L x d

        # Context awareness
        t = torch.transpose(posEmbed, 0, 1)
        pool1 = torch.transpose(torch.mm(t, self.PA), 0, 1)                     # L x d

        concat = torch.cat([posEmbed, pool1], 1)                                # L x 2d
        dense1 = torch.mm(concat, self.D1)                                      # L x d

        # Context awarenes for SACA
        t = torch.transpose(dense1, 0, 1)
        pool2 = torch.transpose(torch.mm(t, self.PA), 0, 1)                     # L x d

        # SACA
        concatSACA = torch.cat([dense1, pool2], 1)                              # L x 2d
        attention = torch.mm(concatSACA, self.Att)                              # L x d
        weighted = dense1 * attention                                           # L x d
        t = torch.transpose(weighted, 0, 1)
        attSum = torch.transpose(torch.mm(t, self.SA), 0, 1)                    # L x d
        SACA = torch.cat([self.zero, attSum[:-1]], 0)                           # L x d

        # Output
        out = torch.mm(SACA, self.D2)
        out = out[-1].unsqueeze(0)

        if self.activation is not None:
            out = self.activation(out)

        return out
