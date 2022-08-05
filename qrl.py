# Q-Learning agent
import argparse
from timeit import default_timer as timer
import datetime
import numpy as np

from main import BOARD_ROWS, BOARD_COLS, State, MILESTONES, OBJ


class QLAgent:
    def __init__(self, start_state, win_state, lr=0.2, exp_rate=0.5, decay_gamma=0.9, obj=True):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.win_state = win_state
        self.start_state = start_state
        self.obj = obj
        print("Constructing agent with start", start_state, "win state", win_state)
        self.State = State(state=self.start_state, win_state=self.win_state, obj=self.obj)
        self.isEnd = self.State.isEnd
        self.decay_gamma = decay_gamma
        self.lr = lr
        self.exp_rate = exp_rate

        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0

    def chooseAction(self):
        mx_nxt_reward = 0
        action = ""
        # Choose action with the highest expected value.
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def showPolicyValues(self):
        directions = {"left": "<", "right": ">", "up": "^", "down": "v"}
        for i in range(0, BOARD_ROWS):
            print('-------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                q_vals = self.Q_values[(i, j)]
                st = directions[max(q_vals, key=q_vals.get)]
                if self.obj and (i, j) in OBJ:
                    st = "----"
                if (i, j) == self.win_state:
                    st = "*"
                if (i, j) == self.start_state:
                    st += " (s)"

                out += str(st).ljust(6) + ' | '
            print(out)
        print('-------------------------------------------------------------------------------------------')
        print()
    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate

                # Set all actions of the last state as the current reward which is either +1 or -1.
                # Helps to converge faster.
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                #print("Game End Reward", reward)
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1  # End of episode, increase counter until reached max number of episodes.
            else:
                action = self.chooseAction()
                # append trace
                self.states.append((self.State.state, action))
                # print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                # print("nxt state", self.State.state)
                # print("---------------------")
                self.isEnd = self.State.isEnd

    def reset(self):
        self.states = []
        self.State = State(self.start_state, self.win_state, self.obj)
        self.isEnd = self.State.isEnd

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(position, self.win_state, self.obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path-planning using value iteration.")
    parser.add_argument("-o", "--obj", type=bool, default=False,
                        help="Whether using red blocks or not. Default: False.")
    parser.add_argument("-l", "--learning_rate", default=0.2, type=float, help="Learning rate")
    parser.add_argument("-e", "--exp_rate", default=0.5, type=float,
                        help="Exploration rate.")
    parser.add_argument("-eps", "--episodes", default=100, type=float,
                        help="Number of episodes to train for.")

    # TODO: Implement timeout functionality.
    parser.add_argument("-to", "--timeout", default=2, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()

    start = 0
    end = 1
    total_length = 0
    obj = ARGS.obj
    p = []
    names = ["home",
             "grocery store",
             "school",
             "construction site",
             "hospital"]
    while end < len(MILESTONES):
        # Repeatedly train rl agent to reach each milestone.

        ag = QLAgent(start_state=MILESTONES[start], win_state=MILESTONES[end],
                   lr=ARGS.learning_rate, exp_rate=ARGS.exp_rate, obj=obj)
        print("Inital Q values ... \n")
        # TODO; get arrow with max q value, put on grid.
        #print(ag.Q_values)
        print("Start: ", datetime.datetime.now())  # Do not delete
        start_time = timer()
        ag.play(ARGS.episodes)
        end_time = timer()
        print("Latest Q values computed using q learning:")
        #print(ag.Q_values)
        print("End: ", datetime.datetime.now())  # Do not delete
        print("Time taken               ", end_time - start_time)

        # TODO: How will the policies look?
        # print("Policy for the above state values:")
        #
        # # The policy as an array of (state, action) pairs.
        # s = ag.getPolicy()
        #
        # # *** Extract the actions from the policy. This will be used in integration into UI. ***
        # actions = [a for _, a in s]
        #
        # total_length += len(s) - 1
        print(names[start], "to", names[end], "after", total_length, "blocks.")
        ag.showPolicyValues()
        start += 1
        end += 1

    # TODO: Get total length of states.
    # print("Total length: ", total_length, "blocks.")