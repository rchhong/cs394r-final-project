# cs394r-final-project

## Write-Up
Write up can be found [here](https://www.overleaf.com/project/624b384a670a21fbbabaf362)

## Acknowledgements
Utilized a modified version of the wordle gym found [here](https://github.com/zach-lawless/gym-wordle)

## Installation
TODO: Write installation script

```shell
# Install the gym
cd gym-wordle
pip install .

# Train Actor Critic
python -m actor_critic.train_a2c --words_dir ./gym-wordle/gym_wordle/data/5_words.txt
```