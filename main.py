from simple_grid_world import GridWorldEnv
from q_learning_agent import QLearningAgent
from maze_world import MazeWorld
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # environment initialisation
    env = GridWorldEnv(10)
    env2 = MazeWorld()

    # agent initialisation
    q_learning_agent = QLearningAgent(env2)

    # training variables
    num_episodes = 500
    max_steps = 100

    # We reset the environment for the first time
    observation, info = env2.reset(first_reset=True, return_info=True)
    state = env2.state_table.index(list(observation['agent']))

    # training
    q_learning_agent.learn(num_episodes, max_steps, env2, state)

    print(f"Training completed over {num_episodes} episodes")
    print(q_learning_agent.qtable)
    input("Press Enter to watch trained agent...")

    while (True):
        # watch trained agent
        observation = env2.reset()
        state = env2.state_table.index(list(observation['agent']))
        print(state)
        done = False
        rewards = 0

        for s in range(max_steps):

            print(f"TRAINED AGENT")
            print("Step {}".format(s + 1))
            action = np.argmax(q_learning_agent.qtable[state, :])
            print("action :", action)
            observation, reward, done, info = env2.step(action)
            new_state = env2.state_table.index(list(observation['agent']))
            print(new_state)
            rewards += reward
            env2.render()
            print(f"score: {rewards}")
            state = new_state

            if done == True:
                break
    env2.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
