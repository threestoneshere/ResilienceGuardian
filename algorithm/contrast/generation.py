from algorithm.contrast.utils import Config
import os
from algorithm.contrast.paths import cache_path_public, train_generation_path_public, test_generation_path_public
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pickle
import tqdm
import json
import random
import hashlib
import sklearn.cluster as sc


logger = logging.getLogger('ClassTrast')


def build_dataset(ds_name, path_to_config):
    config = DatasetConfig(ds_name, path_to_config)
    Dataset(config)
    
    
class DatasetConfig(Config):
    def __init__(self,
                 ds_name,
                 path_to_config, 
                 time,
                 section='dataset' 
                 ):
        super(DatasetConfig, self).__init__(ds_name, path_to_config, section)
        keys_necessary = ['k', 'ds_path', 'period', 'duration', 'interval', 'n_period_exp', 'n_period_normal', 'n_period_check', 'smooth', 'omega']
        keys_optional = {'omega': 1.2, 'intensity_default': 1.5}
        self.time = time
        for key in keys_necessary:
            assert key in self.__dict__.keys()
        
        for key, value in keys_optional.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)
        self.ds_info = pd.read_csv(os.path.join(self.ds_path, 'ds_info.csv')) 

    
    @property
    def ds_md5(self):
        m = hashlib.md5()
        path = os.path.join(self.ds_path, 'labels.json')
        
        with open(path, 'rb') as fin:
            while True:
                data = fin.read(4096)
                if not data:
                    break
                m.update(data)

        return m.hexdigest()
    
'''
def normalize_series(values):
    thres = np.nanpercentile(values, 95)
    mean = np.nanmean(values[values <= thres])
    std = np.nanstd(values[values <= thres])
    if std == 0:
        return np.ones(len(values))
    values_normalized = (values - mean) / std
    return values_normalized'''


def normalize_series(values):
    q = np.percentile(values, 95)
    if q > 0:
        values = list(np.array(values) / q)
    values = list((np.array(values) - 0.5) * 2)
    return values

def moving_average(data, window=4, min_periods=1):
    df = pd.Series(data)
    moving_avg = df.rolling(window=window, min_periods=min_periods).mean()
    moving_avg = moving_avg.values.flatten()
    
    return np.array(moving_avg)


def deal_with_deviated_points(seq, intensity_default=1.5, big_mul_default=50, anomalous_mul_default=20.0):
    big_points = list(seq > np.average(seq) + big_mul_default * intensity_default)
    small_points = list(seq < np.average(seq) - big_mul_default * intensity_default)
    for big_point in big_points:
        copy_point = big_point
        while(copy_point in big_points):
            copy_point -= 1
        seq[big_point] = seq[copy_point] + intensity_default * anomalous_mul_default
    
    for small_point in small_points:
        copy_point = small_point
        while(copy_point in small_points):
            copy_point -= 1
        seq[small_point] = seq[copy_point] - intensity_default * anomalous_mul_default
    return seq


class Dataset:
    def __init__(self,
                 config: DatasetConfig,
                 exp_lists):
        self.ds_path = config.ds_path
        self.ds_info = config.ds_info
        self.ds_name = config.ds_name
        self.config = config
        self.exp_lists = exp_lists
        self.test_generation_path = os.path.join(test_generation_path_public, config.time)
        self.process_normal_data().data
        self.build_dataset()
        
        
        
    def process_normal_data(self):
        return DataGeneration(self.config, 'normal')
        
    def build_dataset(self):
        for exp in self.exp_lists:
            for target in os.listdir(os.path.join(self.ds_path, exp)):
                data_v1 = DataGeneration(self.config, os.path.join(exp, target), 'v1')
                data_v2 = DataGeneration(self.config, os.path.join(exp, target), 'v2')
                self.generate_test_data(data_v1, data_v2)
                    
    
    def generate_test_data(self, data_v1, data_v2):
        test_data = []
        test_save_dir = os.path.join(self.test_generation_path, self.config.ds_name, data_v1.subpath)
        os.makedirs(test_save_dir, exist_ok=True)
        test_data_cache = os.path.join(test_save_dir, 'test.pkl')
        if os.path.isfile(test_data_cache):
            logger.info(f'Testing data for {self.config.ds_name}-{data_v1.subpath} has been generated.')
            # return
        
        for idx in range(len(self.ds_info)):
            test_instance = []
            kpi_name = self.ds_info.iloc[idx]['kpi'][:-4]
            for i in range(self.config.n_period_exp):
                seq1 = np.array(data_v1.seqs_origin[idx][i])
                seq2 = np.array(data_v2.seqs_origin[idx][i])
                
                pair = np.array([seq1, seq2]).transpose()
                test_instance.append((pair, -1, -1))
                
                for seq_period in data_v1.seqs_period[idx][i]:
                    seq_period = np.array(seq_period)
                    pair = np.array([seq1, seq_period]).transpose()
                    test_instance.append((pair, 1, i))
                seq_local = np.array(data_v1.seqs_local[idx][i])
                pair = np.array([seq1, seq_local]).transpose()
                test_instance.append((pair, 1, i))
                
                for seq_period in data_v2.seqs_period[idx][i]:
                    seq_period = np.array(seq_period)
                    pair = np.array([seq2, seq_period]).transpose()
                    test_instance.append((pair, 2, i))
                seq_local = np.array(data_v2.seqs_local[idx][i])
                pair = np.array([seq2, seq_local]).transpose()
                test_instance.append((pair, 2, i))
            test_data.append(test_instance)
            
        with open(test_data_cache, 'wb') as fout:
            pickle.dump(test_data, fout)
        logger.info(f'Generation has been finished.')


def remove_extreme_data(data, top_n=0.05, mode='deviation-mean'):
   
    if mode == 'extreme':
        # 去除最大最小极值
        half_length = int(len(data) * top_n / 2)
        sorted_args = np.argsort(data)
        data[sorted_args[:half_length]] = np.nan
        data[sorted_args[-half_length:]] = np.nan
    elif mode == 'deviation-mean':
        # 去除偏离均值最大的5%的数据
        length = int(len(data) * top_n)
        mean = np.mean(data)
        abs_distance_mean = np.abs(data-mean)
        sorted_args = np.argsort(abs_distance_mean)
        data[sorted_args[-length:]] = np.nan
    data = deal_nan(data)
    return data


def deal_nan(data):
    x = np.arange(len(data))
    if sum(np.isnan(data)) == data.shape:
        np.nan_to_num(data, copy=False, nan=1)
    else:
        nan_index = np.where(np.isnan(data))[0]
        non_nan_index_x = np.setdiff1d(x, nan_index)
        non_nan_index_y = data[non_nan_index_x]
        data[nan_index] = np.interp(nan_index, non_nan_index_x, non_nan_index_y)
    return data

            
class DataGeneration:
    def __init__(self,
                 config: DatasetConfig,
                 subpath,
                 version='normal', 
                 normal_data=None):
        self.version = version
        self.intensity_default = config.intensity_default
        self.ds_info = config.ds_info
        self.n_kpi = len(self.ds_info)
        self.period = int(config.period / config.interval)
        self.version = version
        self.subpath = subpath
        self.config = config
        self.seqs_origin = []
        self.omega = int(config.omega * config.duration / config.interval)
        self.start = int((config.period - config.duration) / 2 / config.interval)
        self.cache_path = os.path.join(cache_path_public, config.time)
        self.train_generation_path = os.path.join(train_generation_path_public, config.time)
            
        normal_seqs_origin_path = os.path.join(self.cache_path, config.ds_name, 'seqs_origin.pkl')
        self.normal_seqs_origin = None
        self.std = [0] * self.n_kpi
        if version != 'normal':
            assert version in ['v1', 'v2']
            self.seqs_local = []
            self.seqs_period = []
            self.n_period = config.n_period_exp
            self.ds_path = os.path.join(config.ds_path, subpath, version)
            with open(normal_seqs_origin_path, 'rb') as f:
                self.normal_seqs_origin = pickle.load(f)
            
        else:
            self.n_period = config.n_period_normal
            self.ds_path = os.path.join(config.ds_path, subpath)
            # Since the omega of seqs is subjected to DatasetConfig, the origin file should be regenerated when building a dataset.
            if os.path.isfile(normal_seqs_origin_path):
                os.remove(normal_seqs_origin_path)
        
        cache_save_dir = os.path.join(self.cache_path, config.ds_name, subpath)
        os.makedirs(cache_save_dir, exist_ok=True)
        data_cache = os.path.join(cache_save_dir, f'{version}.pkl')
        
        if not os.path.isfile(data_cache):
            logger.info(f'Building dataset: {config.ds_name}-{subpath}...')
            self.data = [None] * self.n_kpi
            def read_csv(id, file, remove):
                df = pd.read_csv(file)
                values = np.array(df['value'])
                if remove:
                    values = remove_extreme_data(values)
                values_normalized = normalize_series(values)
                df['value'] = values_normalized
                if config.smooth != 0:
                    smoothing_window = int(len(df['value']) * config.smooth)
                    df['value'] = df['value'].ewm(span=smoothing_window).mean().values.flatten()
                df['value'] = moving_average(values_normalized)
                return (id, df)
            
            executor = ThreadPoolExecutor(20)
            tasks = []
            if version == 'normal':
                remove = True
            else:
                remove = False
            for _, row in self.ds_info.iterrows():
                id = row['id']
                file = os.path.join(self.ds_path, row['kpi'])
                tasks.append(executor.submit(read_csv, id, file, remove))

            for future in as_completed(tasks):
                id, df = future.result()
                self.data[id] = df
            with open(data_cache, 'wb') as fout:
                pickle.dump(self.data, fout)
            logger.info(f'{config.ds_name}-{subpath} has been built.')
        
        else:
            logger.info(f'Loading data of {config.ds_name}-{subpath} from cache...')
            with open(data_cache, 'rb') as fin:
                self.data = pickle.load(fin)
            logger.info('Loaded.')
        
        kpi_std_cache_file = os.path.join(self.cache_path, config.ds_name, 'kpi_std.json')
        if os.path.isfile(kpi_std_cache_file):
            with open(kpi_std_cache_file) as f:
                kpi_std_cache = json.load(f)
        else:
            kpi_std_cache = {}
        
        for i in range(self.n_kpi):
            kpi_name = self.ds_info.iloc[i]['kpi'][:-4]
            if kpi_name not in kpi_std_cache:
                df = self.data[i]
                df['timestamp'] = df['timestamp'] % self.period
                df.sort_values(by='timestamp', inplace=True)
                arr = np.array(df)
                start_idx = 0
                std_list = []
                while start_idx < len(arr):
                    end_idx = start_idx
                    while end_idx + 1 < len(arr) and arr[end_idx + 1, 0] == arr[start_idx, 0]:
                        end_idx += 1
                    std = np.nanstd(arr[start_idx:end_idx+1, 1])
                    
                    std_list.append(std)
                    start_idx = end_idx + 1
                std_avg = float(np.mean(std_list))
                # kpi_std_cache[kpi_name] = min(std_avg, 1.5)
                kpi_std_cache[kpi_name] = std_avg
            std_avg = kpi_std_cache[kpi_name]    
            self.std[i] = std_avg
    
        if version == 'normal':
            assert len(self.ds_info) == len(kpi_std_cache.keys())
            
            with open(kpi_std_cache_file, 'w') as f:
                json.dump(kpi_std_cache, f, indent=2)
            
            self.get_seqs_origin()
            with open(normal_seqs_origin_path, 'wb') as fout:
                pickle.dump(self.seqs_origin, fout)
            self.normal_seqs_origin = self.seqs_origin 
            self.generate_train_data()
            
        else:
            self.get_seqs_origin()
            self.generate_train_data()
            
            
    def generate_train_data(self):
        train_save_dir = os.path.join(self.train_generation_path, self.config.ds_name, self.subpath)
        os.makedirs(train_save_dir, exist_ok=True)
        train_data_cache = os.path.join(train_save_dir, f'{self.version}.pkl')
        if os.path.isfile(train_data_cache):
            logger.info(f'Training data for {self.config.ds_name}-{self.subpath} has been generated.')
            return
        logger.info(f'Begin to generate training data for {self.config.ds_name}-{self.subpath}...')
        train_data = []
        for idx in range(len(self.ds_info)):
            for _ in range(1000):
                train_data.append(self.get_one_pair(idx))
                train_data.append(self.get_one_pair(idx, normal=False))
        with open(train_data_cache, 'wb') as fout:
            pickle.dump(train_data, fout)
        logger.info(f'Generation has been finished.')
        
    
    def scale_and_average_seq(self, idx, seq):
        seq =  np.array(pd.DataFrame(seq).ewm(span=int(len(seq) * 0.02)).mean().values.flatten())
        intensity = self.std[idx]
        intensity_mul = self.intensity_default / intensity if intensity != 0 else 1.0
        return seq * intensity_mul
        
        
    def get_seqs_origin(self):
        for idx in range(self.n_kpi):
            seqs_origin = []
            for i_seq in range(self.n_period):
                seq_origin = deal_with_deviated_points(self.scale_and_average_seq(idx, np.array(self.data[idx].iloc[self.start + i_seq * self.period:self.start + self.omega + i_seq * self.period]['value'])))
                seqs_origin.append(seq_origin)
            self.seqs_origin.append(seqs_origin)
        if self.version in ['v1', 'v2']:
            for idx in range(self.n_kpi):
                seqs_local = []
                for i_seq in range(self.n_period):
                    seq_local = deal_with_deviated_points(self.scale_and_average_seq(idx, np.array(self.data[idx].iloc[self.start - self.omega + i_seq * self.period:self.start + i_seq * self.period]['value'])))
                    # seq_local2 = np.array(self.data[idx].iloc[self.start + self.omega + i_seq * self.period:self.start + 2 * self.omega + i_seq * self.period]['value'])
                    seqs_local.append(seq_local)
                self.seqs_local.append(seqs_local)
            for idx in range(self.n_kpi):
                seqs_period = []
                normal_seqs_origin = self.normal_seqs_origin[idx]
                normal_seqs_origin_length = len(normal_seqs_origin)
                for i_seq in range(self.n_period):
                    seq_period = []
                    for _ in range(self.config.n_period_check):
                        seq_period.append(deal_with_deviated_points(self.scale_and_average_seq(idx, np.array(normal_seqs_origin[random.randint(0, normal_seqs_origin_length - 1)]))))
                    seqs_period.append(seq_period)
                self.seqs_period.append(seqs_period)
                
    
    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.percentile(np.abs(np.array(a) - np.array(b)), 80))
        
        
    def get_one_pair(self, idx, normal=True):
        seqs_origin = self.seqs_origin[idx]
        normal_seqs_origin = self.normal_seqs_origin[idx]
        
        def get_one_seq(normal):
            seq = seqs_origin[random.randint(0,self.n_period - 1)]
            if normal:
                if np.random.rand() < 0.5:
                    seq = self._add_noises(seq, True)
            else:
                if np.random.rand() < 0.6:
                    seq = self._add_noises(seq, False)
                else:
                    new_seq = None
                    try_times = 10
                    
                    # Make sure the new seq is dissimilar enough to the current seq
                    while new_seq is None:
                        normal_seqs_origin_length = len(normal_seqs_origin)
                        try_seq = normal_seqs_origin[random.randint(0, normal_seqs_origin_length - 1)]
                        if np.random.rand() < 0.5:
                            try_seq = self._add_noises(try_seq, True)
                        if np.random.rand() < 0.5:
                            if self._distance(try_seq, seq) < min(self.intensity_default * 2, 0.5):
                                try_times -= 1
                                if try_times > 0:
                                    continue
                            else:
                                new_seq = try_seq
                                continue
                        new_seq = self._add_noises(try_seq, False)
                    seq = new_seq
            if len(seq[np.isnan(seq)]) > 0:
                print(1)
            return seq
        
        
        seq1 = get_one_seq(True)
        seq2 = get_one_seq(normal)
        if normal:
            seq1 = np.append(seq1, [0])
            seq2 = np.append(seq2, [0])
        else:
            seq1 = np.append(seq1, [0])
            seq2 = np.append(seq2, [1])
        pair = np.array([seq1, seq2]).transpose()
        
        return pair
        

    def _add_noises(self, d: np.ndarray, normal) -> np.ndarray:
        intensity = self.intensity_default
        def level_shift(d: np.ndarray, intensity: float) -> np.ndarray:
            t = np.random.uniform(intensity / 2, intensity)
            if np.random.rand() < 0.5:
                t *= -1
            return d + t
        def relative_level_shift(d: np.ndarray, intensity: float) -> np.ndarray:
            t = np.random.uniform(intensity / 2, intensity)
            if np.random.rand() < 0.5:
                t *= -1
            t = 1 - t
            return (d + 1) * t - 1
        def gaussian_noise(d: np.ndarray, intensity: float) -> np.ndarray:
            noise = np.random.normal(0, intensity, d.shape)
            return d + noise
        def transient_noise(d: np.ndarray, intensity: float) -> np.ndarray:
            pos = np.random.randint(0, len(d))
            t = np.random.uniform(-intensity * 5, intensity * 5)
            d_ = d.copy()
            d_[pos] += t
            return d_
        def ramp(d: np.ndarray, intensity: float) -> np.ndarray:
            pos = np.random.randint(0, len(d))
            t = np.random.uniform(intensity / 2, intensity)
            if np.random.rand() < 0.5:
                t *= -1
            d_ = d.copy()

            mode = np.random.randint(0, 10)
            if mode == 0:
                d_[:pos] += t
            elif mode == 1:
                d_[pos:] += t
            else:
                pos_ = np.random.randint(0, len(d))
                if pos > pos_:
                    pos, pos_ = pos_, pos
                d_[pos:pos_] += t
            return d_
        def steady_change(d: np.ndarray, intensity: float) -> np.ndarray:
            sgn = np.random.choice((-1, 1))
            pos = np.random.randint(1, len(d) // 4 * 3)
            t = np.random.uniform(intensity / (len(d)-pos), intensity / (len(d)-pos) * 2)

            noise = np.zeros(d.shape)
            for i in range(pos, len(d)):
                noise[i] = noise[i-1] + np.random.normal(t / 2, t)
            noise *= sgn
            return d + noise

        # Modules are not exclusive. They are selected by probability.
        funcs = []
        if np.random.rand() < 0.4:
            funcs.append([level_shift, intensity])
        if np.random.rand() < 0.4:
            funcs.append([relative_level_shift, intensity])
        if np.random.rand() < 0.2:
            funcs.append([transient_noise, intensity])
        if np.random.rand() < 0.8 or len(funcs) == 0:
            funcs.append([gaussian_noise, intensity])
        if not normal:
        
            if np.random.rand() < 0.3:
                funcs.append([level_shift, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.3:
                funcs.append([relative_level_shift, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.3:
                funcs.append([gaussian_noise, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.2:
                funcs.append([ramp, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.2:
                funcs.append([steady_change, intensity * np.random.uniform(3, 10)])
            
            if not funcs:
                funcs.append([steady_change, intensity * np.random.uniform(3, 10)])
        for f, f_intensity in funcs:
            d = f(d, f_intensity)
        return d


    

            
            
            
        
        
        