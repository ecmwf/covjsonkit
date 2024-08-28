import json
import os
from os.path import dirname

from conflator import Conflator

from .config import CovjsonKitConfig

param_dir = os.getenv("PARAM_DIR", "ecmwf")
conf = Conflator(app_name="covjsonkit", model=CovjsonKitConfig).load()
param_dir = conf.param_db


def get_param_from_db(param_id):
    """
    import requests
    url = f"https://codes.ecmwf.int/parameter-database/api/v1/param/?format=json&search={param_id}"
    response = requests.get(url)
    if response.status_code == 200:
        for param in response.json():
            if param['id'] == param_id:
                return param
    else:
        raise BaseException(f"Failed to get parameter from database: {response.status_code}")
    """
    try:
        param_id = int(param_id)
    except BaseException:
        param_id = get_param_id_from_db(param_id)

    param_path = os.path.join(dirname(__file__), f"data/{param_dir}/param.json")
    with open(param_path) as f:
        params = json.load(f)
    return params[str(param_id)]


def get_param_id_from_db(param_id):
    param_id_path = os.path.join(dirname(__file__), f"data/{param_dir}/param_id.json")
    with open(param_id_path) as f:
        param_ids = json.load(f)
    return param_ids[str(param_id)]


def get_unit_from_db(unit_id):
    unit_path = os.path.join(dirname(__file__), f"data/{param_dir}/unit.json")
    with open(unit_path) as f:
        units = json.load(f)
    return units[str(unit_id)]


def get_param_ids(conf):
    param_dir = conf.param_db
    param_id_path = os.path.join(dirname(__file__), f"data/{param_dir}/param_id.json")
    with open(param_id_path) as f:
        param_ids = json.load(f)
    return param_ids


def get_params(conf):
    param_dir = conf.param_db
    param_path = os.path.join(dirname(__file__), f"data/{param_dir}/param.json")
    with open(param_path) as f:
        params = json.load(f)
    return params


def get_units(conf):
    param_dir = conf.param_db
    unit_path = os.path.join(dirname(__file__), f"data/{param_dir}/unit.json")
    with open(unit_path) as f:
        units = json.load(f)
    return units
