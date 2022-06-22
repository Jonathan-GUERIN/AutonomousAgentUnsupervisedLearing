import gym
import gym_examples
from q_learning_agent import QLearningAgent
import numpy as np
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = gym.make('gym_examples/GridWorld-v0')
    env.action_space.seed(42)

    state_table = [[i, j] for i in range(env.size) for j in range(env.size)]

    # initialisation de l'agent
    q_learning_agent = QLearningAgent(env.nb_states, env.nb_actions, env.action_space)

    # variables d'entrainement
    num_episodes = 1000
    max_steps = 120

    observation, info = env.reset(seed=42, return_info=True)

    for episode in range(num_episodes):

        q_learning_agent.state = state_table.index(list(observation['agent']))

        for s in range(max_steps):

            action = q_learning_agent.choose_action() # choose an action according to the explore/exploit policy

            observation, reward, done, info = env.step(action)
            new_state = state_table.index(list(observation['agent']))

            # Q-learning algorithm
            q_learning_agent.learn(action, new_state, reward)

            # if done, finish episode
            if done == True:
                # On reset l'environnement
                observation = env.reset()
                break

        # Decrease epsilon
        q_learning_agent.decrease_epsilon(episode)

    print(f"Training completed over {num_episodes} episodes")
    print(q_learning_agent.qtable)
    input("Press Enter to watch trained agent...")
    # watch trained agent
    observation = env.reset()
    state = state_table.index(list(observation['agent']))
    print(state)
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))
        action = np.argmax(q_learning_agent.qtable[state, :])
        print("action :", action)
        observation, reward, done, info = env.step(action)
        new_state = state_table.index(list(observation['agent']))
        print(new_state)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break
    env.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
