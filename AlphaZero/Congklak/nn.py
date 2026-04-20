#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from parameters import Parameters

class BasicBlock(nn.Module):
    def __init__(self, num_filters):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        r = x
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h += r
        h = F.relu(h)
        return h

class Net(nn.Module):
    def __init__(self):
        self.params = Parameters()
        super(Net, self).__init__()
        # Input channels: (k_boards * 2) + 1 = 3
        self.conv1 = nn.Conv2d(self.params.input_channels, self.params.num_filters, 3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(self.params.num_filters)

        self.blocks = self._make_layer(self.params.num_res, self.params.num_filters)

        self.conv_p = nn.Conv2d(self.params.num_filters, self.params.num_filters_p, 1, stride=1)
        self.bn_p   = nn.BatchNorm2d(self.params.num_filters_p)
        # Board size is 2x8. Action size is 7.
        self.fc_p = nn.Linear(self.params.num_filters_p * self.params.board_x * self.params.board_y, self.params.action_size)

        self.conv_v = nn.Conv2d(self.params.num_filters, self.params.num_filters_v, 1, stride=1)
        self.bn_v   = nn.BatchNorm2d(self.params.num_filters_v)
        self.fc_v1 = nn.Linear(self.params.num_filters_v * self.params.board_x * self.params.board_y, 256)
        self.bn_v1 = nn.BatchNorm1d(256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, x):
        # x is (batch, channels, 2, 8)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.blocks(h)

        h_p = F.relu(self.bn_p(self.conv_p(h)))
        h_p = h_p.view(-1, self.params.num_filters_p * self.params.board_x * self.params.board_y)
        h_p = self.fc_p(h_p)
        p = F.log_softmax(h_p, dim=1)

        h_v = F.relu(self.bn_v(self.conv_v(h)))
        h_v = h_v.view(-1, self.params.num_filters_v * self.params.board_x * self.params.board_y)
        h_v = F.relu(self.bn_v1(self.fc_v1(h_v)))
        h_v = self.fc_v2(h_v)
        v = torch.tanh(h_v)

        return p, v

    def _make_layer(self, blocks, num_filters):
        layers = []
        for _ in range(blocks):
            layers.append(BasicBlock(num_filters))
        return nn.Sequential(*layers)

class NNetWrapper:
    def __init__(self, params = Parameters(), device = None):
        self.params = params
        self.net = Net()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Final safety check: if 'cuda' requested but not available, fallback to 'cpu'
        if 'cuda' in self.device and not torch.cuda.is_available():
            self.device = 'cpu'
        
        # Performance: move to device once
        self.to_device()

    def to_device(self):
        self.net.to(torch.device(self.device))

    def predict(self, states):
        device = torch.device(self.device)
        # Performance: removed self.net.to(device) from here
        board = torch.Tensor(states).to(device)
        if len(board.shape) == 3:
            board = board.unsqueeze(0) # add batch dimension

        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(board)

        return torch.exp(pi).data.to('cpu').numpy()[0], v.item()

    def train(self, training_board, training_prob, training_v):
        device = torch.device(self.device)
        self.net.to(device)

        training_board = torch.Tensor(training_board)
        training_prob = torch.Tensor(training_prob)
        training_v = torch.Tensor(training_v)

        ds_train = torch.utils.data.TensorDataset(training_board, training_prob, training_v)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size = self.params.batch_size, shuffle=True, num_workers = 1, pin_memory = False)
        optimizer = optim.SGD(self.net.parameters(), lr = self.params.lam, weight_decay = self.params.weight_decay, momentum = self.params.momentum)

        total_pi_loss = 0
        total_v_loss = 0
        count = 0

        for epoch in range(self.params.epochs):
            self.net.train()
            for batch_idx, (boards, pis, vs) in enumerate(train_loader):
                boards, pis, vs = boards.to(device), pis.to(device), vs.to(device)
                out_pis, out_vs = self.net(boards)
                l_pi = self.loss_pi(pis, out_pis)
                l_v = self.loss_v(vs, out_vs)
                total_l = l_pi + l_v
                optimizer.zero_grad()
                total_l.backward()
                optimizer.step()

                total_pi_loss += l_pi.item()
                total_v_loss += l_v.item()
                count += 1

        self.net.to('cpu')
        if 'cuda' in self.device:
            torch.cuda.empty_cache()

        return (total_pi_loss / count if count > 0 else 0, 
                total_v_loss / count if count > 0 else 0)


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2)/targets.size()[0]

    def save_checkpoint(self, i):
        # Always save to 'checkpoint.model' so it acts as the "latest" reference
        torch.save({'state_dict': self.net.state_dict(), 'iteration': i}, "checkpoint.model")

        # If it's a checkpoint interval, also save a numbered backup
        if i % self.params.checkpoint_interval == 0:
            torch.save({'state_dict': self.net.state_dict(), 'iteration': i}, f"checkpoint_{i}.model")

    def load_checkpoint(self, i = None):
        filename = "checkpoint.model" if i is None else f"checkpoint_{i}.model"
        checkpoint = torch.load(filename, map_location='cpu', weights_only=True)
        # Support both old format (plain state_dict) and new format (dict with iteration)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
            self.net.eval()
            return checkpoint.get('iteration', 0)
        else:
            # Old format: checkpoint is the state_dict itself
            self.net.load_state_dict(checkpoint)
            self.net.eval()
            return 0
