import json
import os
from os.path import dirname


def get_param_from_db(param_id):
    """
    import requests
    url = f"https://codes.ecmwf.int/parameter-database/api/v1/param/?format=json&searcg={param_id}"
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

    param_path = os.path.join(dirname(__file__), "data/param.json")
    with open(param_path) as f:
        params = json.load(f)
    return params[str(param_id)]


def get_param_id_from_db(param_id):
    param_id_path = os.path.join(dirname(__file__), "data/param_id.json")
    with open(param_id_path) as f:
        param_ids = json.load(f)
    return param_ids[str(param_id)]


def get_unit_from_db(unit_id):
    unit_path = os.path.join(dirname(__file__), "data/unit.json")
    with open(unit_path) as f:
        units = json.load(f)
    return units[str(unit_id)]
