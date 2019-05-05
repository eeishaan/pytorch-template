Light weight experiment framework for quick prototyping in pytorch.

# Definitions
- Experiment: A collection of net, optimizer and loss function. Defines how net is trained and tested.
- Model: A class sub-classing `torch.nn.Module`
- Param file: A YAML file containing experiment details such as model name, optimizer name, loss name and experiment name. [Click here](skeletal/param/sample.yml) to view a sample file.

# A word on design
- For training a neural network in pytorch, we need
  - A Model: This is the neural network
  - An optimizer such as SGD
  - A loss function (a.k.a criterion)
- These are all independent components which are combined together to train the model.
- For our case, the abstraction that combines all of the above components is referred to as an Experiment.
- An Experiment handles all the details when it comes to training and evaluating a model.
- It also handles checkpointing and resume training code.
- Currently the base experiment class is incapable of training multiple networks at once. For example, it can't be used to train a GAN. But, can be extended pretty easily to do so.
- The Experiment abstraction allows us to reuse a lot of boilerplate code while reducing the chances of an error.

# How to add new algorithms
- Make a new model and place it inside `skeletal/model` directory.
- Register model class with the model factory in `skeletal.factory.model`. You can either make a new factory class based on the provided interface or add entries to existing one.
- This will allow users to train model via command line.
- It is not mandatory to add a model.
- Next we add a new experiment. This is where the training logic goes.
- You can either:
  - Use `skeletal.experiment.BaseExperiment` without any changes.
  - Subclass the `skeletal.experiment.BaseExperiment` template experiment to make a new one. Place it under `skeletal/experiment` directory.
- `BaseExperiment`class provides a lot of boiler plate code and reduces code duplication. It also provides a lot of hooks that can be used to write custom algorithms.
- Register the chosen experiment in experiment factory at `skeletal.factory.experiment`.
- You can add new custom optimizer and loss functions by adding entries to their factories. Currently all pytorch optimizers and loss functions can be used just by writing their class name in the param file.
- Construct a param file based on [sample param file](skeletal/param/sample.yml).

# How to trigger training
- Once the above steps are done, you need to construct the training command.
- Example command that works out of box
    ```
    python3 -m skeletal train --model cnn --params skeletal/param/sample.yml
    ```
This is a python3 code base. Compatibility with python2 is not guaranteed. Tested on Ubuntu 18.04.
