# cs394r-final-project

## Write-Up
Write up can be found [here](https://www.overleaf.com/project/624b384a670a21fbbabaf362)

## Acknowledgements
Utilized a modified version of the wordle gym found [here](https://github.com/zach-lawless/gym-wordle)

## Dependencies
These can be found in the file `requirements.txt`, but the purpose of each package will be outlined here

* gym - Utilized this to write the Wordle environment
* colorama - Required for visualizing the Wordle environment
* setuptools - Used in setting up the gym
* numpy - Used for fast computation
* torch - Utilized to write and train neural networks


## Installation

```shell
# Install dependencies
pip install -r requirements.txt

# Install the gym
cd gym-wordle
pip install .

cd ..
mkdir models
```

## REINFORCE with Baseline
To run the training for REINFORCE with baseline, run the following command from the project root directory.
```shell
python3 -m reinforce.train_reinforce --words_dir ./gym-wordle/gym_wordle/data/5_words.txt --log_dir log
```

TensorBoard logging can be accessed in `log/REINFORCE`.

## Actor-Critic Methods
To run the training for A2C, run the following command from the project root directory.
```shell
python3 -m actor_critic.train_a2c --words_dir ./gym-wordle/gym_wordle/data/5_words.txt --log_dir log
```

TensorBoard logging can be accessed in `log/A2C`.
