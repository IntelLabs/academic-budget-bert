# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import random
from itertools import product

import yaml


def get_yaml(file_name):
    with open(file_name, "r") as stream:
        try:
            ym = yaml.safe_load(stream)
            return ym
        except yaml.YAMLError as e:
            print(e)


def get_run_id():
    t = datetime.datetime.now()
    time_str = t.strftime("%Y%m%d%H%M%S")
    random_num = random.randint(10000, 100000)
    return f"{time_str}-{random_num}"


def add_run_id_per_command(params_combinations_named):
    for comb in params_combinations_named:
        comb["current_run_id"] = get_run_id()
    return params_combinations_named


def get_hyper_param_combinations_grid(parameters_json):
    params = parameters_json["hyperparameters"]
    map_index_name = list(params.keys())
    all_params_list = [param_values for _, param_values in params.items()]

    params_combinations = list(product(*all_params_list))
    params_combinations_named = [
        {map_index_name[i]: value for i, value in enumerate(comb)} for comb in params_combinations
    ]
    params_combinations_named = add_run_id_per_command(params_combinations_named)
    return params_combinations_named


def get_hyper_param_combinations(parameters_json, search_type="grid"):
    cases = {"grid": get_hyper_param_combinations_grid}

    how_to_get_hyper_param_combinations = cases["grid"]

    if search_type in cases:
        how_to_get_hyper_param_combinations = cases[search_type]

    return how_to_get_hyper_param_combinations(parameters_json)


def add_param(key, value):
    if type(value) == bool:
        return f"--{key}"
    return f"--{key} {value}"


def get_command_from_params(param_list):
    return " ".join([add_param(k, v) for k, v in param_list.items()])


def append_command(command, addition):
    return f"{command} {addition}"


def add_default_params(parameters_json, job_name):
    parameters_json["default_parameters"]["job_name"] = job_name
    return parameters_json


def get_command_per_combination(command_init, parameters_json, params_combinations_named):
    all_commands = []
    command_default = get_command_from_params(parameters_json["default_parameters"])

    for comb in params_combinations_named:
        command_current = f"{command_init}"
        command_current = append_command(command_current, get_command_from_params(comb))
        command_current = append_command(command_current, command_default)
        all_commands.append(command_current)
    return all_commands


def create_experiments(command_init, param_file, job_name, search_type="grid"):
    parameters_json = get_yaml(param_file)
    parameters_json = add_default_params(parameters_json, job_name)
    params_combinations_named = get_hyper_param_combinations(parameters_json, search_type)
    all_commands = get_command_per_combination(
        command_init, parameters_json, params_combinations_named
    )
    for command in all_commands:
        print(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", help="Hyperparameter and configuration yaml", required=True)
    parser.add_argument("--job_name", help="job name", default="bert_large_experiment")
    parser.add_argument(
        "--init_cmd",
        help="initialization command (deepspeed or python directly)",
        default="deepspeed run_pretraining.py",
    )
    parser.add_argument("--search_type", help="hyperparameter search method", default="grid")
    args = parser.parse_args()

    create_experiments(args.init_cmd, args.param_file, args.job_name, args.search_type)
