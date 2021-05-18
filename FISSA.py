import torch
from torch import nn
from torch.nn import functional
import math

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class FISSA(torch.nn.Module):
    def __init__(self, I, L, d, B=5, activation=None):
        super().__init__()

        # Constants
        self.ones = torch.ones((1, L))
        self.delta = torch.tril(torch.ones((L, L)))

        # Save Hyperparameters
        self.I = I
        self.L = L
        self.d = d
        self.B = B
        self.activation = activation

        # Learned parameters
        self.M = nn.Parameter(torch.Tensor(self.I, self.d))
        self.P = nn.Parameter(torch.Tensor(self.L, self.d))
        self.qS = nn.Parameter(torch.Tensor(1, self.d))
        self.Wg = nn.Parameter(torch.Tensor(3 * self.d, 1))
        self.bg = nn.Parameter(torch.Tensor(1, 1))
        self.Q = nn.Parameter(torch.Tensor(self.d, self.d))
        self.K = nn.Parameter(torch.Tensor(self.d, self.d))
        self.V = nn.Parameter(torch.Tensor(self.d, self.d))
        self.Wk = nn.Parameter(torch.Tensor(self.d, self.d))
        self.Wv = nn.Parameter(torch.Tensor(self.d, self.d))
        self.W1 = nn.Parameter(torch.Tensor(self.d, self.d))
        self.W2 = nn.Parameter(torch.Tensor(self.d, self.d))
        self.b1 = nn.Parameter(torch.Tensor(1, self.d))
        self.b2 = nn.Parameter(torch.Tensor(1, self.d))

        # Parameter initialization
        print("Initializing model with hyperparameters:" +
              "\nI=" + str(self.I) +
              "\nL=" + str(self.L) +
              "\nd=" + str(self.d) +
              "\nB=" + str(self.B) + "\n")
        for p in self.parameters():
            nn.init.kaiming_normal_(p, a=math.sqrt(5))

    def forward(self, S, candidate):
        # Encoded input
        E = torch.mm(S, self.M)

        # Global Representation
        y = torch.mm(functional.softmax(torch.mm(self.qS, torch.transpose(torch.mm(E, self.Wk), 0, 1)), dim=-1), torch.mm(E, self.Wv))
        Y = torch.cat([functional.dropout(y) for _ in range(0, self.L)])                                                             # L x d

        # Local Representation
        X = E + self.P
        for _ in range(0, self.B):

            # Self Attention Layer
            XQ = torch.mm(X, self.Q)
            XK = torch.mm(X, self.K)
            XV = torch.mm(X, self.V)
            SAL = torch.mm(torch.mm(functional.softmax(torch.mm(XQ, torch.transpose(XK, 0, 1))/math.sqrt(self.d), dim=-1), self.delta), XV)

            # Feed Forward Layer
            XW = torch.mm(SAL, self.W1)
            oneT = torch.transpose(self.ones, 0, 1)
            X = torch.mm(functional.relu(XW + torch.mm(oneT, self.b1)), self.W2) + torch.mm(oneT, self.b2)                              # L x d

        # Gating
        m = torch.mm(candidate, self.M)                                                                                                 # 1 x d
        mRep = m.repeat((self.L, 1))                                                                                                    # L x d

        conc = torch.cat((mRep, Y, E), dim=1)
        bRep = self.bg.repeat((self.L, 1))
        g = torch.sigmoid(torch.mm(conc, self.Wg) + bRep)                                                                               # L x 1
        g = g.repeat((1, self.d))                                                                                                       # L x d

        Z = X * g + Y * (1 - g)                                                                                                         # L x d

        # Output
        R = torch.mm(Z, torch.transpose(m, 0, 1))                                                                                       # L x 1

        if self.activation is not None:
            R = self.activation(R)

        return R
