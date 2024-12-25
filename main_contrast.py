from algorithm.contrast.experiment import begin_experiments
from algorithm.contrast.model import ContrastConfig
from algorithm.contrast.generation import DatasetConfig
import os
from copy import deepcopy
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


'''for i in range(10):
    ds_name = 'testbed'
    path_to_config = 'algorithm/contrast/configs'
    timestamp = str(int(time.time())) + 'check_train_wo_big_test_wo_big'
    dataset_config = DatasetConfig(ds_name, path_to_config, timestamp)
    omega = int(1.2 * dataset_config.duration / dataset_config.interval)
    model_config = ContrastConfig(ds_name, path_to_config, timestamp, input_len=omega, k=dataset_config.k)
    begin_experiments('testbed', path_to_config, generation=True, train_flag=True, test_flag=True, timestamp=timestamp,  model_config=model_config, exp_lists=['exp6', 'exp7', 'exp8', 'exp9', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14', 'exp15', 'exp16', 'exp17', 'exp18'], run_name=f'no_cate')'''


# compute f1 with the existed model
for ts in os.listdir('algorithm/contrast/result_diff'):
    if 'train_wo_big_test_wo_big' not in ts:
        continue
    ds_name = 'testbed'
    path_to_config = 'algorithm/contrast/configs'
    timestamp = ts
    dataset_config = DatasetConfig(ds_name, path_to_config, timestamp)
    omega = int(1.2 * dataset_config.duration / dataset_config.interval)
    model_config = ContrastConfig(ds_name, path_to_config, timestamp, input_len=omega, k=dataset_config.k)
    begin_experiments('testbed', path_to_config, generation=False, train_flag=False, test_flag=True, timestamp=timestamp,  model_config=model_config, exp_lists=['exp6', 'exp7', 'exp8', 'exp9', 'exp10', 'exp11', 'exp12', 'exp13', 'exp14', 'exp15', 'exp16', 'exp17', 'exp18'], run_name=f'no_cate')


"""ds_name = 'testbed'
path_to_config = 'algorithm/contrast/configs'
timestamp = '1687719636check_train_big_test_wo_big'
dataset_config = DatasetConfig(ds_name, path_to_config, timestamp)
omega = int(1.2 * dataset_config.duration / dataset_config.interval)
model_config = ContrastConfig(ds_name, path_to_config, timestamp, input_len=omega, k=dataset_config.k)
begin_experiments('testbed', path_to_config, generation=True, train_flag=False, test_flag=True, timestamp=timestamp,  model_config=model_config, exp_lists=['exp8'], run_name=f'no_cate')"""

