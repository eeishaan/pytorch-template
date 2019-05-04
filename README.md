Light weight experiment framework for quick prototyping in pytorch.

# Definitions
- Experiment: A collection of net, optimizer and loss function. Defines how net is trained and tested.
- Model: A class sub-classing `torch.nn.Module`
- Param file: A YAML file containing experiment details such as model name, optimizer name, loss name and experiment name. [Click here](skeletal/param/sample.yml) to view a sample file.

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
- You can add new custom optimizer and loss functions by adding entries to their factories.
- Construct a param file based on [sample param file](skeletal/param/sample.yml).

# How to trigger training
- Once the above steps are done, you need to construct the training command.
- Example command that works out of box
    ```
    python3 -m skeletal train --model cnn --params skeletal/param/sample.yml
    ```
