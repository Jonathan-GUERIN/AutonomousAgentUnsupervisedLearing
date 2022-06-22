import gym
import gym_examples
from q_learning_agent import QLearningAgent
import numpy as np
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = gym.make('gym_examples/GridWorld-v0')
    env.action_space.seed(42)

    # initialisation de l'agent
    q_learning_agent = QLearningAgent(env)

    # training variables
    num_episodes = 50
    max_steps = 120

    # We reset the environment for the first time
    observation, info = env.reset(first_reset=True, return_info=True)
    state = env.state_table.index(list(observation['agent']))

    # training
    q_learning_agent.learn(num_episodes, max_steps, env, state)

    print(f"Training completed over {num_episodes} episodes")
    print(q_learning_agent.qtable)
    input("Press Enter to watch trained agent...")

    # watch trained agent
    observation = env.reset()
    state = env.state_table.index(list(observation['agent']))
    print(state)
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))
        action = np.argmax(q_learning_agent.qtable[state, :])
        print("action :", action)
        observation, reward, done, info = env.step(action)
        new_state = env.state_table.index(list(observation['agent']))
        print(new_state)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break
    env.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
