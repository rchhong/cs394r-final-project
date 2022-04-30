import gym
import gym_wordle

from gym_wordle.exceptions import InvalidWordException

env = gym.make('Wordle-v0')

obs = env.reset()

done = False
while not done:
    while True:
        try:
            human = input()
            if (len (human) != 5):
                pass
            
            act = []
            for char in human.lower():
                act.append (ord(char) - 97)

            obs, reward, done, _ = env.step(act)
            print (obs.shape, obs)
            print ("rewards (green - 2, yellow - 1, grey - 0): {}".format(reward))
            
            break
        except InvalidWordException:
            pass

    env.render()