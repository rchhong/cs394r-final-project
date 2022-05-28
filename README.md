# Wordle RL Agent

## Write-Up
Write up can be found [here](writeup.pdf)

## Acknowledgements
Utilized a modified version of the wordle gym found [here](https://github.com/zach-lawless/gym-wordle)

Utilizes 3B1B's entropy agent as a comparison found [here](https://github.com/3b1b/videos/tree/master/_2022/wordle)

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
WARNING: MAKE SURE TO CHANGE THE BOOLEAN AT THE TOP OF THE FILE IF YOU ARE NOT RUNNING ON THE VERSION WITH LIMITED TARGETS

To run the training for REINFORCE with baseline, run the following command from the project root directory.
```shell
python3 -m reinforce.train_reinforce --words_dir ./gym-wordle/gym_wordle/data/5_words.txt --log_dir log
```

If running the variant with restricted targets.
```shell
python3 -m reinforce.train_reinforce --words_dir ./gym-wordle/gym_wordle/data/5_words.txt --log_dir log -n 60000 --possible_solutions_dir ./gym-wordle/gym_wordle/data/possible_solutions.txt -c
```

TensorBoard logging can be accessed in `log/REINFORCE`.

## Actor-Critic Methods
To run the training for A2C, run the following command from the project root directory.
```shell
python3 -m actor_critic.train_a2c --words_dir ./gym-wordle/gym_wordle/data/5_words.txt --log_dir log
```

TensorBoard logging can be accessed in `log/A2C`.

This may not work with CUDA, did not test this with it.
