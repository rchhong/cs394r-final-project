import gym
import gym_wordle

from gym_wordle.exceptions import InvalidWordException

env = gym.make('CartPole-v0')

obs = env.reset()
done = False
while not done:
    while True:
        try:
            human = input()
            if (human == "a"):
                action = 1
            else:
                action = 0
            # if (len (human) != 5):
            #     pass
            
            # act = []
            # for char in human.lower():
            #     act.append (ord(char) - 97)
            # make a random guess
            # act = env.action_space.sample()
            # print (act)
            # take a step
            obs, reward, done, _ = env.step(action)
            print (obs)
            # print ("rewards (green - 2, yellow - 1, grey - 0): {}".format(reward))
            # print (obs['state'])
            break
        except InvalidWordException:
            pass

    env.render()