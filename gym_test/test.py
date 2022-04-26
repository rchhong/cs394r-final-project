import gym
import gym_wordle

from gym_wordle.exceptions import InvalidWordException

env = gym.make('Wordle-v0')

obs = env.reset()
env.hidden_word = [19, 0, 5, 5, 24]
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
            # make a random guess
            # act = env.action_space.sample()
            # print (act)
            # take a step
            obs, reward, done, _ = env.step(act)
            if(act == env.hidden_word):
                print("bark bark bark")

            print ("rewards (green - 2, yellow - 1, grey - 0): {}".format(reward))
            print (obs)
            break
        except InvalidWordException:
            pass

    env.render()