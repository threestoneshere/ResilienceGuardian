from algorithm.contrast.utils import setup_logging
from algorithm.contrast.generation import Dataset, DatasetConfig
import logging
from algorithm.contrast.model import Contrast, ContrastConfig, DataLoader
from algorithm.contrast.paths import train_generation_path_public, test_generation_path_public, result_path_public, log_path_public
import os
import pickle
from collections import defaultdict
import json
import torch
import numpy as np
import pandas as pd
import time


logger = setup_logging()

def begin_experiments(ds_name, path_to_config, generation=True, train_flag=True, test_flag=True, timestamp=None, model_config=None, run_name='default', exp_lists=None, by_exp=True, model_ts_name=None):
    experiments = Experiments(ds_name, path_to_config, generation, train_flag, test_flag, timestamp, model_config, run_name, exp_lists, by_exp, model_ts_name)
    experiments.run()
        
                
class Experiments:
    def __init__(self, ds_name, path_to_config, generation, train_flag, test_flag, timestamp, model_config, run_name, exp_lists, by_exp, model_ts_name):
        if timestamp is None:
            timestamp = int(time.time())
        self.timestamp = str(timestamp)
        os.makedirs(log_path_public, exist_ok=True)
        hdlr = logging.FileHandler(os.path.join(log_path_public, f'{self.timestamp}-{run_name}.log'))
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        self.result_path = os.path.join(result_path_public, self.timestamp)
        
        dataset_config = DatasetConfig(ds_name, path_to_config, self.timestamp)
        ds_info = dataset_config.ds_info
        if exp_lists is None:
            exp_lists = []
            for dir in os.listdir(dataset_config.ds_path):
                if dir in ['normal', 'ds_info.csv']:
                    continue
                exp_lists.append(dir)
        
        if generation:
            Dataset(dataset_config, exp_lists)
        omega = int(dataset_config.omega * dataset_config.duration / dataset_config.interval)
        if model_config is None:
            model_config = ContrastConfig(ds_name, path_to_config, self.timestamp, input_len=omega, k=dataset_config.k)
        
        train_generation_dir = os.path.join(train_generation_path_public, self.timestamp, ds_name)
        test_generation_dir = os.path.join(test_generation_path_public, self.timestamp, ds_name)
        self.by_exp = by_exp
        if self.by_exp:
            self.exps = []
            for exp in exp_lists:
                exp_path = os.path.join(train_generation_dir, exp)
                for target in os.listdir(exp_path):
                    train_data_path = os.path.join(train_generation_dir, exp, target)
                    test_data_path = os.path.join(test_generation_dir, exp, target)
                    label_path = os.path.join(dataset_config.ds_path, exp, target, 'labels.json')
                    self.exps.append(Experiment(ds_info, model_config, label_path, train_data_path, test_data_path, f'{ds_name}-{exp}-{target}-{run_name}', train_flag, test_flag, exp_lists, model_ts_name,  dataset_config.n_period_check))
        else:
        # Use one model for all experiments
            label_path = dataset_config.ds_path
            train_data_path = train_generation_dir
            test_data_path = test_generation_dir
            self.exp = Experiment(ds_info, model_config, label_path, train_data_path, test_data_path, f'{ds_name}-{run_name}', train_flag, test_flag, exp_lists, model_ts_name, dataset_config.n_period_check)
    
    def run(self):
        if self.by_exp:
            for exp in self.exps:
                exp.run()
        else:
            self.exp.run_all()
                

class Experiment:
    def __init__(self, ds_info, model_config, label_path, train_data_path, test_data_path, save_name, train_flag, test_flag, exp_lists, model_ts_name, n_period_check):
        self.exp_lists = exp_lists
        self.n_period_check = n_period_check
        self.model = Contrast(model_config, save_name, model_ts_name)
        self.input_dim = model_config.input_dim
        self.label_path = label_path
        self.batch_size = model_config.batch_size
        self.input_dim = model_config.input_dim
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.save_name = save_name
        self.ds_info = ds_info
        self.train_flag = train_flag
        self.test_flag = test_flag
        self.result_path = os.path.join(result_path_public, model_config.time)
        
        
    def run(self):  
        if self.train_flag:
            self.train()
        else:
            if not os.path.isfile(self.model.save_path):
                print(self.model.save_path)
                #self.train()
                return
        if self.test_flag:
            self.test()
        
    
    def train(self):
        with open(os.path.join(os.path.join(self.train_data_path, '..', '..', 'normal'), 'normal.pkl'), 'rb') as f0:
            train_data_normal = pickle.load(f0)
        with open(os.path.join(self.train_data_path, 'v1.pkl'), 'rb') as f1:
            train_data_v1 = pickle.load(f1)
        with open(os.path.join(self.train_data_path, 'v2.pkl'), 'rb') as f2:
            train_data_v2 = pickle.load(f2)
        
        logger.info(f'Start training: {self.model.model_save_name}...')
        self.model.train(DataLoader(train_data_normal + train_data_v1 + train_data_v2, self.batch_size, self.input_dim))
        logger.info(f'Finished: {self.model.model_save_name}...')
            
    
    def test(self):
        with open(self.label_path) as f:
            labels = json.load(f) 
        probs = {}
        with open(os.path.join(self.test_data_path, 'test.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        m = self.model
        for idx in range(len(test_data)):
            kpi_name = self.ds_info.iloc[idx]['kpi'][:-4]
            probs[kpi_name] = defaultdict(list)
            test_instance = test_data[idx]
            normal_v1_cache = defaultdict(list)
            normal_v2_cache = defaultdict(list)
       
            
            for ins in test_instance:
                pair = torch.Tensor(np.array(np.expand_dims(ins[0], axis=0), dtype='float64')).view(1, -1, self.input_dim)
                if ins[1] == -1:
                    probs[kpi_name]['comparation'].append(float(m.test(pair)[0][1]))
                elif ins[1] == 1:
                    normal_v1_cache[ins[2]].append(float(m.test(pair)[0][1]))
                elif ins[1] == 2:
                    normal_v2_cache[ins[2]].append(float(m.test(pair)[0][1]))
           
            normal_v1_cache = {pair[0]: pair[1] for pair in sorted(normal_v1_cache.items(), key=lambda item:int(item[0]))}
            normal_v2_cache = {pair[0]: pair[1] for pair in sorted(normal_v2_cache.items(), key=lambda item:int(item[0]))}
            
            for key in normal_v1_cache.keys():
                probs[kpi_name]['normal_v1'].append(np.min(np.array(normal_v1_cache[key])))
                probs[kpi_name]['normal_v2'].append(np.min(np.array(normal_v2_cache[key])))
        
        result_keys = []
        result_mean_com_probs = []
        result_mean_norm_v1_probs = []
    
        result_mean_norm_v2_probs = []
    
        result_com_labels = []
        result_norm_v1_labels = []
        result_norm_v2_labels = []
        result_better_labels = []
        for k, v in probs.items():
            result_keys.append(k)
            result_mean_com_probs.append(np.mean(np.array(v['comparation'])))
            result_mean_norm_v1_probs.append(np.mean(np.array(v['normal_v1'])))
            result_mean_norm_v2_probs.append(np.mean(np.array(v['normal_v2'])))
            
            result_com_labels.append(labels[k]['diff'])
            result_norm_v1_labels.append(labels[k]['anomaly'][0])
            result_norm_v2_labels.append(labels[k]['anomaly'][1])
            
            result_better_labels.append(labels[k]['better'])
            
        result = pd.DataFrame({'kpi': result_keys, 
                               'diff': result_mean_com_probs, 
                               'anomaly0': result_mean_norm_v1_probs, 
                            
                               'anomaly1': result_mean_norm_v2_probs, 
                               
                               'diff_label': result_com_labels,
                               'anomaly0_label': result_norm_v1_labels,
                               'anomaly1_label': result_norm_v2_labels, 
                               'better_label': result_better_labels
        })
        return result