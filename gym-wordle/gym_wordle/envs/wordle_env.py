import gym
from gym import spaces
import numpy as np
import pkg_resources
import random
from typing import Optional
import colorama
from colorama import Fore
from colorama import Style

from gym_wordle.exceptions import InvalidWordException

colorama.init(autoreset=True)

# global game variables
GAME_LENGTH = 6
WORD_LENGTH = 5

# load words and then encode
filename = pkg_resources.resource_filename(
    'gym_wordle',
    'data/5_words.txt'
)

def encodeToStr(encoding):
    string = ""
    for enc in encoding:
        string += chr(ord('a') + enc)
    return string

def strToEncode(lines):
    encoding = []
    for line in lines:
        assert len(line.strip()) == 5  # Must contain 5-letter words for now
        encoding.append(tuple(ord(char) - 97 for char in line.strip()))
    return encoding


with open(filename, "r") as f:
    WORDS = strToEncode(f.readlines())


class WordleEnv(gym.Env):
    """
    Simple Wordle Environment

    Wordle is a guessing game where the player has 6 guesses to guess the
    5 letter hidden word. After each guess, the player gets feedback on the
    board regarding the word guessed. For each character in the guessed word:
        * if the character is not in the hidden word, the character is
          grayed out (encoded as 0 in the environment)
        * if the character is in the hidden word but not in correct
          location, the character is yellowed out (encoded as 1 in the
          environment)
        * if the character is in the hidden word and in the correct
          location, the character is greened out (encoded as 2 in the
          environment)

    The player continues to guess until they have either guessed the correct
    hidden word, or they have run out of guesses.

    The environment is structured in the following way:
        * Action Space: the action space is a length 5 MulitDiscrete where valid values
          are [0, 25], corresponding to characters [a, z].
        * Observation Space: the observation space is dict consisting of
          two objects:
          - board: The board is 6x5 Box corresponding to the history of
            guesses. At the start of the game, the board is filled entirely
            with -1 values, indicating no guess has been made. As the player
            guesses words, the rows will fill up with values in the range
            [0, 2] indicating whether the characters are missing in the
            hidden word, in the incorrect position, or in the correct position
			based on the most recent guess.
          - alphabet: the alphabet is a length 26 Box corresponding to the guess status
            for each letter in the alaphabet. As the start, all values are -1, as no letter
            has been used in a guess. As the player guesses words, the letters in the
            alphabet will change to values in the range [0, 2] indicating whether the
            characters are missing in the hidden word, in the incorrect position,
            or in the correct position.
    """

    def __init__(self):
        super(WordleEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([26] * WORD_LENGTH)
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=2, shape=(GAME_LENGTH, WORD_LENGTH), dtype=int),
            'alphabet': spaces.Box(low=-1, high=2, shape=(26,), dtype=int),
            'state' : spaces.Box(low=0, high=2, shape=(26,5), dtype=int)
        })
        self.guesses = []
        self.state_space = np.zeros((26, 5, 3), dtype=np.int0)
        self.state_space[:,:,1] = 1

    def step(self, action):
        assert self.action_space.contains(action)

        # Action must be a valid word
        if not tuple(action) in WORDS:
            raise InvalidWordException(encodeToStr(action) + " is not a valid word.")

        # update game board and alphabet tracking
        board_row_idx = GAME_LENGTH - self.guesses_left

        left = {x : 0 for x in range (26)}


        for idx, char in enumerate(action):

            if self.hidden_word[idx] == char:
                encoding = 2

                #Set everything else to 0 except correct character 
                self.state_space[:, idx, 0] = 1
                self.state_space[char, idx, :] = 0
                self.state_space[char, idx, 2] = 1
            else:
                encoding = 0
                left[self.hidden_word[idx]] += 1

            self.board[board_row_idx, idx] = encoding
            self.alphabet[char] = encoding


        for idx, char in enumerate(action):
            if char in self.hidden_word and left[char] >= 1:
                encoding = 1
                left[char] -= 1
                # state_space[char, idx] = 1
            else:
                encoding = 0

            if (self.board[board_row_idx, idx] != 2):
                self.board[board_row_idx, idx] = encoding
                self.alphabet[char] = encoding
                res = np.where(self.state_space[char, :, 1] == 1, encoding, np.argmax (self.state_space[char], axis=1))
                b = np.zeros((5, 3))
                b[np.arange(5), res] = 1
                self.state_space[char, :] = b

        # update guesses remaining tracker
        self.guesses_left -= 1

        # if (str (action) in [str(x) for x in self.guesses]):
        #     reward = -100
        #     if self.guesses_left > 0:
        #         done = False
        #     else:
        #         done = True
        #     return self._get_obs(), reward, done, {}

        # update previous guesses made
        self.guesses.append(action)


        reward = 0

        # reward = np.sum(self.board[board_row_idx, :])
        
        # # check to see if game is over
        # if all(self.board[board_row_idx, :] == 2):
        #     done = True
        # else:
        #     if self.guesses_left > 0:
        #         done = False
        #     else:
        #         done = True

        if all(self.board[board_row_idx, :] == 2):
            reward = 10.0
            done = True
        else:
            if self.guesses_left > 0:
                reward = -1.0
                done = False
            else:
                reward = -10.0
                done = True


        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        
        # for i in range (26):
        #     # print (self.state_space[i])
        #     print (chr(i + 97), np.argmax (self.state_space[i], axis=1))
        # for board_idx in range (len(self.guesses)):
        #     for idx in range (5):
        return_state = np.zeros((26 + 5 * 26 * 3 + 6))
        return_state[0:26] = np.where(self.alphabet == -1, 0, 1)
        return_state[26:-6] = self.state_space.flatten()
        return_state[6 * 26 + self.guesses_left - 1] = 1
        
        return return_state

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        self.hidden_word = random.choice(WORDS)
        self.guesses_left = GAME_LENGTH
        self.board = np.negative(
            np.ones(shape=(GAME_LENGTH, WORD_LENGTH), dtype=int))
        self.alphabet = np.negative(np.ones(shape=(26,), dtype=int))
        self.guesses = []
        self.state_space = np.zeros((26, 5, 3), dtype=np.int0)
        self.state_space[:,:,1] = 1
        return self._get_obs()

    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('###################################################')
        for i in range(len(self.guesses)):
            for j in range(WORD_LENGTH):
                letter = chr(ord('a') + self.guesses[i][j])
                if self.board[i][j] == 0:
                    print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 1:
                    print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 2:
                    print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            print()
        print()

        for i in range(len(self.alphabet)):
            letter = chr(ord('a') + i)
            if self.alphabet[i] == 0:
                print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 1:
                print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 2:
                print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == -1:
                print(letter + " ", end='')
        print()
        print('###################################################')
        print()
