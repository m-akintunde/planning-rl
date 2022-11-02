import numpy as np

from agents.agent import Agent
from state import State
from utils import BOARD_ROWS, BOARD_COLS


class QLAgent(Agent):
    def __init__(self, start_state, win_state, cm, lr=0.2, exp_rate=0.5, decay_gamma=0.9, obj=True):
        print("Constructing q-learning (non-deterministic) agent with start state", start_state, "win state", win_state)
        super().__init__(start_state, win_state, cm, lr, exp_rate, obj)

        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0

        self.isEnd = self.State.isEnd
        self.decay_gamma = decay_gamma

    def chooseAction(self):
        mx_nxt_reward = float('-inf')
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

    def choosePolicyAction(self):
        return max(self.Q_values[self.State.state], key=self.Q_values[self.State.state].get)

    def showPolicyValues(self, states, objs, emergency_objs):
        directions = {"left": "<", "right": ">", "up": "^", "down": "v"}
        for i in range(0, BOARD_ROWS):
            print('-------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                q_vals = self.Q_values[(i, j)]
                st = directions[max(q_vals, key=q_vals.get)]
                if (i, j) in emergency_objs or self.obj and (i, j) in objs:
                    st += " --"
                if (i, j) == self.win_state:
                    st += " *"
                if (i, j) == self.start_state:
                    st += " (s)"

                out += str(st).ljust(6) + ' | '
            print(out)
        print('-------------------------------------------------------------------------------------------')
        print()

    def showValues(self, states, objs, emergency_objs):
        for i in range(0, BOARD_ROWS):
            print('-------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                q_vals = self.Q_values[(i, j)]
                st = max(q_vals, key=q_vals.get)
                out += str(st).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # If at the end of game back propagate reward
            if self.State.isEnd:
                # back propagate

                # Set all actions of the last state as the current reward (must be 1 since no fail state).
                # Helps to "converge faster".
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                #print("Game End Reward", reward)
                self.reset()
                i += 1  # End of episode, increase counter until reached max number of episodes.
                # print("Episode:", i)
            else:
                action = self.chooseAction()
                # append trace
                self.states.append((self.State.state, action))
                s = self.State.state

                # print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                r = self.State.giveReward()

                best_next_action = max(self.Q_values[self.State.state], key=self.Q_values[self.State.state].get)
                td_target = r + self.decay_gamma * self.Q_values[self.State.state][best_next_action]
                td_delta = td_target - self.Q_values[s][action]
                # print(td_delta)
                self.Q_values[s][action] += self.lr * td_delta

                # mark is end
                # self.Q_values[s][action] = r + max(self.Q_values[self.State.state].values())
                self.State.isEndFunc()
                # print("nxt state", self.State.state)
                # print("---------------------")
                self.isEnd = self.State.isEnd

    def reset(self):
        self.states = []
        self.State = State(self.start_state, self.win_state, False, self.cm, self.obj)
        self.isEnd = self.State.isEnd

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(position, self.win_state, False, self.cm, self.obj)

    def chooseNondetPolicyAction(self):
        mx_nxt_reward = 0
        action = ""
        for a in self.actions:
            # if the action is deterministic (does this apply here?)
            # Get state corresponding to next position.
            nxt_pos = self.State.nxtPosition(a)
            if nxt_pos == self.State.state or (len(self.states) > 0 and nxt_pos in [s[0] for s in self.states]):
                nxt_reward = 0
            else:
                nxt_reward = self.Q_values[nxt_pos][a]
            if nxt_reward > mx_nxt_reward:
                action = a
                mx_nxt_reward = nxt_reward
        return action

    def getNondetPolicy(self):
        # "Reset" with initial state at the beginning of path.
        self.State = State(self.start_state, self.win_state, self.determine, self.cm, self.obj)
        self.states = [(self.State.state, "*")]
        while not self.State.isEnd:
            action = self.chooseNondetPolicyAction()
            # append trace
            self.states[-1] = (self.states[-1][0], action)
            self.states.append((self.State.nxtPolicyPosition(action), "*"))
            print("current position {} action {}".format(self.State.state, action))
            # by taking the action, it reaches the next state
            self.State = self.takeAction(action)
            # mark is end
            self.State.isEndFunc()
            print("nxt state", self.State.state)
            print("---------------------")
        return self.states

