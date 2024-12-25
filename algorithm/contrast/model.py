from torch import nn, optim
import torch.nn.functional as F
import torch
from algorithm.contrast.utils import timer, Config
import tqdm
import numpy as np
import logging
import os
from algorithm.contrast.paths import ckpt_save_path_public


logger = logging.getLogger('Contrast')


class DataLoader:
    def __init__(self,
                 data: np.ndarray,
                 batch_size: int,
                 input_dim: int
                 ):
        self.data = np.random.permutation(data)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.cache = []
        self.cnt = 0
    
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    @property
    def sample_count(self):
        return len(self.data)
    
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        batch_pair = []
        batch_label = []
        remaining = self.sample_count - self.cnt
        idx = self.cnt // self.batch_size
        
        if remaining > 0:
            current_batch_size = min(remaining, self.batch_size)
            if len(self.cache) > idx:
                batch_pair, batch_label = self.cache[idx]
            else:
                for i in range(self.cnt, self.cnt + current_batch_size):
                    batch_pair.append(self.data[i][:-1])
                    batch_label.append(self.data[i][-1][1])
                    
                batch_pair = torch.Tensor(np.array(batch_pair, dtype=float)).view(current_batch_size, -1, self.input_dim)
                batch_label = torch.Tensor(np.array(batch_label, dtype=int)).view(current_batch_size,)
                self.cache.append((batch_pair, batch_label))
            
            self.cnt += current_batch_size
            return (batch_pair, batch_label)
        
        else:
            self.cnt = 0
            self.data = np.random.permutation(self.data)
            raise StopIteration


class ContrastModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim, input_len, num_layer, dropout):
        super(ContrastModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_len = input_len
        self.fc_dim = fc_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim[0], batch_first=True) #, num_layers=num_layer, dropout=dropout)
        self.normalizer = nn.BatchNorm1d(input_len)
        self.lstm2 = nn.LSTM(hidden_dim[0], hidden_dim[1], batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim[1], fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 2),
            nn.Softmax(dim=1)
        )
    
    
    def forward(self, seq):
        batch_size = seq.shape[0]
        seq = seq.view(batch_size, -1, self.input_dim)
        lstm_out1, _ = self.lstm1(seq)
        lstm_out1_norm = self.normalizer(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1_norm)
        lstm_out2_norm = self.normalizer(lstm_out2)
        fc_in = lstm_out2_norm[:, -1, :]
        fc_out = self.fc(fc_in)
        return fc_out


class ContrastConfig(Config):
    def __init__(self,
                 ds_name,
                 path_to_config, 
                 time,
                 section='model',
                 **kwargs
                 ):
        super(ContrastConfig, self).__init__(ds_name, path_to_config, section, **kwargs)
        self.time = time
        keys_necessary = ['k', 'input_len', 'epoch', 'hidden_dim', 'fc_dim', 'lr', 'dropout', 'num_layer']
        keys_optional = {'batch_size': 1000, 'device': 'cuda:0'}
        
        for key in keys_necessary:
            assert key in self.__dict__.keys()
            
        for key, value in keys_optional.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)
    

class Contrast:
    def __init__(self,
                 config: ContrastConfig,
                 model_save_name,
                 model_ts_name):
        # Model_ts_name being not None indicates a fine-tuning process.
        self.config = config
        self.model = ContrastModel(input_dim=self.config.input_dim,
                             hidden_dim=self.config.hidden_dim,
                             fc_dim=self.config.fc_dim,
                             input_len=self.config.input_len,
                             num_layer=self.config.num_layer,
                             dropout=self.config.dropout
                            )
        self.model.to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model_save_name = model_save_name
        os.makedirs(os.path.join(ckpt_save_path_public, config.time), exist_ok=True)
        self.save_path = os.path.join(ckpt_save_path_public, config.time, f'{self.model_save_name}.ckpt')
        self.model_used_path = None
        if model_ts_name is not None:
            self.model_used_path = os.path.join(ckpt_save_path_public, f'{model_ts_name}.ckpt')
        self.trained = False
        
        
    @timer
    def train(self, dataloader: DataLoader):
        n_epoch = self.config.epoch
        device = self.config.device
        if self.model_used_path is not None:
            self.load()
        model = self.model
        optimizer = self.optimizer
        if self.model_used_path is not None:
            model.lstm1.requires_grad_(False)
            model.lstm2.requires_grad_(False)
            
        for epoch in range(n_epoch):
            loss_all = 0
            # for batch_pair, batch_label in tqdm.tqdm(dataloader):
            for batch_pair, batch_label in dataloader:
                batch_pair = batch_pair.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                fc_out = model(batch_pair)
                loss_function = nn.CrossEntropyLoss()
                loss = loss_function(fc_out, batch_label.type(torch.long))
                loss_all += loss * batch_pair.shape[0]
                loss.backward()
                optimizer.step()
            logger.info(f'epoch {epoch:3d}: loss = {loss_all / (1e-5 + dataloader.sample_count):.6f}')

        self.model = model
        self.trained = True
        self.dump()
    
    
    def test(self, pair):
        if not self.trained:
            self.load()
            self.trained = True
        device = self.config.device
        with torch.no_grad():
            model = self.model
            model.eval()
            pair = pair.to(device)
            prediction = model(pair)
            
        return prediction
    
    
    
    
    def dump(self):
        state = {
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(state, self.save_path)  
        
        
    def load(self):
        ckpt = torch.load(self.save_path)
        self.model.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])


    def inference(self, x_1: np.ndarray, x_2: np.ndarray):
        device = self.config.device
        input_dim = self.config.input_dim
        batch_size = 1 if len(x_1.shape) == 1 else x_1.shape[0]
        x_1 = torch.Tensor(x_1).view((batch_size, -1, input_dim)).to(device)
        x_2 = torch.Tensor(x_2).view((batch_size, -1, input_dim)).to(device)
        with torch.no_grad():
            out_1, out_2 = self.model(x_1, x_2)
            d = F.pairwise_distance(out_1, out_2).item()
            return d


