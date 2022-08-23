import argparse
import numpy as np
from timeit import default_timer as timer
import datetime


# Position of milestones in the order:
# 1. Start state
MILESTONES = [(9, 0), (7, 3), (4, 6), (1, 4), (0, 9)]

BOARD_ROWS = 10
BOARD_COLS = 10

# Positions of red blocks
OBJ = [
    (8, 0)
    , (8, 1)
    , (8, 2)
    , (8, 3)
    , (7, 0)
    , (7, 1)
    , (7, 2)
    , (6, 0)
    , (6, 1)
    , (6, 2)
    , (5, 3)
    , (5, 4)
    , (5, 5)
    , (5, 6)
    , (5, 7)
    , (4, 5)
    , (4, 4)
    , (4, 7)
    , (2, 3)
    , (2, 5)
    , (2, 6)
    , (1, 3)
    , (1, 5)
    , (1, 6)
    , (0, 3)
    , (0, 4)
    , (0, 5)
    , (0, 6)
       ]


# Representation of the gridworld.
class State:
    def __init__(self, state, win_state, determine, obj=True):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.obj = obj
        if self.obj:
            for o in OBJ:
                self.board[o[0], o[1]] = -1
        self.state = state
        self.isEnd = False
        self.determine = determine
        self.win_state = win_state

    def giveReward(self):
        if self.state == self.win_state:
            return 1
        #elif self.state == LOSE_STATE:
        #    return -1
        else:
            return 0

    def isEndFunc(self):
        if self.state == self.win_state:  # or (self.state == LOSE_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        Takes an action, validates the legitimacy of the action, returns the state corresponding to performing that
        action.
        returns: next position
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)

        else:
            # non-deterministic.
            action = self._chooseActionProb(action)
            self.determine = True

            # Call this function again as if in deterministic case.
            nxtState = self.nxtPosition(action)

        # if next state legal
        if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS - 1)):
            if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS - 1)):
                if not self.obj:
                    return nxtState
                # TODO: Treat red blocks as states with negative reward rather than pure obstacle.
                elif self.obj and nxtState not in OBJ:
                    return nxtState
        return self.state

    def nxtPolicyPosition(self, action):
        """
        action: up, down, left, right
        -------------
        return next position
        """
        if action == "up":
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)

        # if next state legal
        if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS - 1)):
            if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS - 1)):
                if not self.obj:
                    return nxtState
                # TODO: Treat red blocks as states with negative reward rather than pure obstacle.
                elif self.obj and nxtState not in OBJ:
                    return nxtState
        return self.state

    # Unused.
    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')



# Value-iteration agent
class Agent:

    def __init__(self, start_state, win_state, lr=0.2, exp_rate=0.5, obj=True):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.win_state = win_state
        self.start_state = start_state
        self.obj = obj
        print("Constructing agent with start", start_state, "win state", win_state)
        self.determine = True
        self.State = State(state=self.start_state, win_state=self.win_state, determine=self.determine, obj=self.obj)
        self.lr = lr

        # Probability of choosing a random action (exploration) rather than choosing an action
        # greedily based on what has already been learned.
        self.exp_rate = exp_rate

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate or \
                all(r == 0 for r in (self.State.nxtPosition(a) for a in self.actions)):
            action = np.random.choice(self.actions)
        else:
            # greedy action, but only if there are some non-zero rewards to greedily choose from.
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def choosePolicyAction(self):
        mx_nxt_reward = 0
        action = ""
        for a in self.actions:
            # if the action is deterministic
            nxt_pos = self.State.nxtPolicyPosition(a)
            if nxt_pos == self.State.state or (len(self.states) > 0 and nxt_pos in [s[0] for s in self.states]):
                nxt_reward = 0
            else:
                nxt_reward = self.state_values[nxt_pos]
            if nxt_reward > mx_nxt_reward:
                action = a
                mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(position, self.win_state, True, self.obj)

    def reset(self):
        self.states = []
        self.State = State(self.start_state, self.win_state, True, self.obj)

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                #print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1  # End of episode, increase counter until reached max number of episodes.
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                #print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                #print("nxt state", self.State.state)
                #print("---------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('-------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                s = self.state_values[(i, j)]
                out += str(s).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    # Extract policy from state rewards.
    def getPolicy(self):
        # "Reset" with initial state at the beginning of path.
        self.State = State(self.start_state, self.win_state, self.determine, self.obj)
        self.states = [(self.State.state, "*")]
        while not self.State.isEnd:
            action = self.choosePolicyAction()
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

    def showValuesStr(self):
        o = ''
        for i in range(0, BOARD_ROWS):
            o += '\n-------------------------------------------------------------------------------------------'
            out = '| '
            for j in range(0, BOARD_COLS):
                s = self.state_values[(i, j)]

                out += str(s).ljust(6) + ' | '
            o += '\n' + out
        o += '\n-------------------------------------------------------------------------------------------'
        return o

    def showPolicyValues(self, states):
        directions = {"left": "<", "right": ">", "up": "^", "down": "v"}
        for i in range(0, BOARD_ROWS):
            print('-------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                st = self.state_values[(i, j)]
                sts = [s for s, _ in states]
                sts_map = dict(states)
                if (i, j) in sts:
                    st = directions.get(sts_map[(i, j)], "*")
                if self.obj and (i, j) in OBJ:
                    st = "----"

                out += str(st).ljust(6) + ' | '
            print(out)
        print('-------------------------------------------------------------------------------------------')
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path-planning using value iteration.")
    parser.add_argument("-o", "--obj", type=bool, default=False,
                        help="Whether using red blocks or not. Default: False.")
    parser.add_argument("-l", "--learning_rate", default=0.2, type=float, help="Learning rate")
    parser.add_argument("-e", "--exp_rate", default=0.5, type=float,
                        help="Exploration rate.")
    parser.add_argument("-eps", "--episodes", default=100, type=float,
                        help="Number of episodes to train for.")
    parser.add_argument("-nd", "--nondet", default=False, type=bool, help="Non-deterministic env or not")

    # TODO: Implement timeout functionality.
    parser.add_argument("-to", "--timeout", default=2, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()

    start = 0
    end = 1
    total_length = 0
    obj = ARGS.obj
    p = []
    names = {1: "grocery store", 2: "school", 3: "construction site", 4: "hospital"}
    while end < len(MILESTONES):
        # Repeatedly train rl agent to reach each milestone.

        ag = Agent(start_state=MILESTONES[start], win_state=MILESTONES[end],
                   lr=ARGS.learning_rate, exp_rate=ARGS.exp_rate, obj=obj)
        print("Start: ", datetime.datetime.now())  # Do not delete
        start_time = timer()
        ag.play(ARGS.episodes)
        print("State values computed using value iteration:")
        ag.showValues()
        end_time = timer()

        print("End: ", datetime.datetime.now())  # Do not delete
        print("Time taken               ", end_time - start_time)
        print("Policy for the above state values:")

        # The policy as an array of (state, action) pairs.
        s = ag.getPolicy()

        # *** Extract the actions from the policy. This will be used in integration into UI. ***
        actions = [a for _, a in s]

        total_length += len(s) - 1
        print(names[end], "after", total_length, "blocks.")
        ag.showPolicyValues(s)
        start += 1
        end += 1

    print("Total length: ", total_length, "blocks.")
