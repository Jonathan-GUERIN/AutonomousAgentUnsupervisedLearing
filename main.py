import gym
import gym_examples
import numpy as np
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = gym.make('gym_examples/GridWorld-v0')
    env.action_space.seed(42)

    # le nbre d'états correspond au nombre de cases du laby pour l'instant
    # A l'avenir il devra correspondre au nombre de cases vides
    state_size = env.size * env.size
    # nbre de déplacements possibles
    action_size = 4

    state_table = [[i, j] for i in range(env.size) for j in range(env.size)]

    # tableau contenant les q-values pour un état S et une action a
    qtable = np.zeros((state_size, action_size))

    # hyperparamètres
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # variables d'entrainement
    num_episodes = 1000
    max_steps = 99

    observation, info = env.reset(seed=42, return_info=True)

    for episode in range(num_episodes):

        state = list(observation['agent'])

        for s in range(max_steps):
            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state_table.index(state), :])

            observation, reward, done, info = env.step(action)
            new_state = list(observation['agent'])

            # Q-learning algorithm
            qtable[state_table.index(state), action] = qtable[state_table.index(state), action] + learning_rate * (
                    reward + discount_rate * np.max(qtable[state_table.index(new_state), :]) - qtable[
                state_table.index(state), action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                print(s)
                # On reset l'environnement
                observation = env.reset()
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")
    print(qtable)
    input("Press Enter to watch trained agent...")
    # watch trained agent
    observation = (env.reset())
    state = list(observation['agent'])
    print(state)
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))
        action = np.argmax(qtable[state_table.index(state), :])
        print("action :", action)
        observation, reward, done, info = env.step(action)
        new_state = list(observation['agent'])
        print(new_state)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break
    env.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
