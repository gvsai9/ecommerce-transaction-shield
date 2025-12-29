import yaml
from src.exception import CustomException
from src.logger import logging
import os,sys
import numpy as np
#import dill
import pickle
import shutil

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)

def update_latest_artifacts(current_artifact_dir: str, latest_dir: str):
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)

    shutil.copytree(current_artifact_dir, latest_dir)
 