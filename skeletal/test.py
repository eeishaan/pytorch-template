#!/usr/bin/env python3

import argparse
import json

import yaml

from skeletal.constants import DATA_ROOT_FOLDER
from skeletal.factory.experiment import SimpleExperimentFactory
from skeletal.factory.model import SimpleModelFactory
from skeletal.utils.data import get_loaders


def get_test_parser(parent=None):
    '''
    Construct argparser for test script
    '''

    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser('test', help='Test pre-trained models')

    parser.add_argument(
        '--model',
        type=str,
        help='Name of model to test',
        choices=SimpleModelFactory.supported_models(),
        required=True,
    )

    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        help='Directory of the test data',
        default=str(DATA_ROOT_FOLDER)
    )

    parser.add_argument(
        '-f', '--model-dir',
        type=str,
        help='Saved model directory',
        required=True
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['valid', 'test', 'train'],
        help='Dataset split to be used',
    )

    parser.add_argument(
        '--params',
        type=str,
        help='Param file location. '
        'For information about param file format refer README.md',
        required=True,
    )

    return parser


def test(args):
    data_dir = args.data_dir
    model_dir = args.model_path
    split = args.split

    # load parameters
    param_file = args.params
    with open(param_file) as fob:
        params = yaml.load(fob)


    # load data
    batch_size = params['batch_size']
    loader = get_loaders(data_dir, split=split, batch_size=batch_size)
    if isinstance(loader, tuple):
        if split == 'valid':
            loader = loader[1]
        else:
            loader = loader[0]
    
    # get model
    model = SimpleModelFactory.make_model(
        args.model, params.get('model_params', {}))

    # set up exp parameters
    experiment_name = params['experiment']
    experiment_params = {
        "experiment_dir": model_dir,
        "model": model,
    }
    experiment_params.update(params.get('exp_params', {}))
    
    # get experiment object
    experiment = SimpleExperimentFactory.make_experiment(
        experiment_name, experiment_params)
    experiment.load_experiment()
    res = experiment.test(loader)
    
    return res

if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args()
    print(test(args))
