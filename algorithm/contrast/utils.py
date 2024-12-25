from time import time
import numpy as np
import configparser
import json
import os
import logging
import sys
import pandas as pd


logger = logging.getLogger('Contrast')


def setup_logging():
    logger = logging.getLogger('Contrast')
    logger.setLevel(logging.INFO)
    
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)
    return logger


def timer(func):
    def deco(*args, **kwargs):
        print(f'\n{func.__name__}() start running...')
        start_time = time()
        res = func(*args, **kwargs)
        end_time = time()
        print(f'{func.__name__}(): {end_time - start_time:.3f}s')
        return res
    return deco



def isfloat(x):
    try:
        x_int = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True
    
    
def isint(x):
    try:
        x_float = float(x)
        x_int = int(x_float)
    except (TypeError, ValueError):
        return False
    else:
        return x_int == x_float
 

def convert_try(x):
    if isint(x):
        return int(float(x))
    elif isfloat(x):
        return float(x)
    else:
        return x
  

class Config:
    def __init__(self, ds_name, path_to_config, section, **kwargs):
        self.path_to_config = path_to_config
        if not os.path.isdir(path_to_config):
            raise ValueError("%s is not a path to a valid directory." % path_to_config)
        self.ds_name = ds_name
        self.sections = ['public', section]
        config_dict = self.get_config_parser()
        for k, v in config_dict.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)
        
            
    def get_config_parser(self):
        config_parser = configparser.ConfigParser()

        config_path = os.path.join(self.path_to_config, self.ds_name + '.conf')
        config_parser.read(config_path)
        
        config_dict = {}
        for section in self.sections:
            config_dict.update(dict(config_parser.items(section)))
        for i, value in config_dict.items():
            if value[0] == '[':
                config_dict[i] = json.loads(value)
            else:
                config_dict[i] = convert_try(value)
        return config_dict




