#!/usr/bin/env python3
import argparse
import datetime
import os

import yaml
from tensorboardX import SummaryWriter

from skeletal.constants import LOG_DIR, SAVED_MODEL_DIR
from skeletal.factory.criterion import SimpleCriterFactory
from skeletal.factory.experiment import SimpleExperimentFactory
from skeletal.factory.model import SimpleModelFactory
from skeletal.factory.optimizer import SimpleOptimFactory
from skeletal.utils.data import get_loaders


def get_train_parser(parent=None):
    '''
    Construct arg parser for train script
    '''
    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser('train', help='Train models')

    parser.add_argument(
        '--model',
        type=str,
        help='Name of model to train',
        choices=SimpleModelFactory.supported_models(),
        required=True,
    )

    parser.add_argument(
        '--params',
        type=str,
        help='Param file location. '
        'For information about param file format refer README.md',
        required=True,
    )

    return parser


def train_model(model_name, params):

    # create experiment directory
    exp_dir = SAVED_MODEL_DIR / params['exp_name'] / datetime.datetime.now()
    os.makedirs(exp_dir, exist_ok=True)

    # make summary writer
    log_dir = str(LOG_DIR / params['exp_name'])
    writer = SummaryWriter(log_dir)
    # write param file information
    writer.add_text('param_file', str(params), 0)

    # load data
    batch_size = params['batch_size']
    train_loader, valid_loader = get_loaders(batch_size=batch_size)

    # get model
    model = SimpleModelFactory.make_model(
        model_name, params.get('model_params', {}))
    
    # get optimizer
    optim_parameters = params['optimizer_params']
    # add model params to optimizer arguments
    optim_parameters['params'] = model.parameters()
    optimizer = SimpleOptimFactory.make_optim(
        params['optimizer'], optim_parameters)

    # get criterion
    crit_parameters = params.get('criterion_params', {})
    criterion = SimpleCriterFactory.make_criter(
        params['criterion'], crit_parameters)


    # set up exp parameters
    experiment_name = params['experiment']
    experiment_params = {
        "experiment_dir": exp_dir,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "summary_writer": writer,
    }
    # add additional experiment parameters
    experiment_params.update(params.get('exp_params', {}))

    # get experiment object
    experiment = SimpleExperimentFactory.make_experiment(
        experiment_name, experiment_params)
    experiment.train(train_loader, valid_loader, params['epochs'])


def train(args):
    param_file = args.params

    # load exp parameters
    with open(param_file) as fob:
        params = yaml.load(fob)

    # train model
    train_model(args.model_name, params)


if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    train(args)
