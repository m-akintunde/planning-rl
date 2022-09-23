import argparse
import numpy as np
from timeit import default_timer as timer
import datetime

from utils import pair_to_int, int_to_pair, MILESTONES, BOARD_ROWS, BOARD_COLS, CM, NAMES


# Representation of the gridworld.
class State:
    def __init__(self, state, win_state, determine, cm, obj=True):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.obj = obj
        self.cm = cm
        self.objs = [int_to_pair(i) for i, j in enumerate(cm) if j == 3]
        if self.obj:
            for o in self.objs:
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
                elif self.obj and nxtState not in self.objs:
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
                elif self.obj and nxtState not in self.objs:
                    return nxtState
        return self.state


# Value-iteration agent
class Agent:

    def __init__(self, start_state, win_state, cm, lr=0.2, exp_rate=0.5, obj=True):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.win_state = win_state
        self.start_state = start_state
        self.obj = obj
        print("Constructing value iteration (deterministic) agent with start", start_state, "win state", win_state)
        self.determine = True
        self.State = State(state=self.start_state, win_state=self.win_state, determine=self.determine, obj=self.obj, cm=cm)
        self.lr = lr
        self.cm = cm

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
        return State(position, self.win_state, True, self.cm, self.obj)

    def reset(self):
        self.states = []
        self.State = State(self.start_state, self.win_state, True, self.cm, self.obj)

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
        self.State = State(self.start_state, self.win_state, self.determine, self.cm, self.obj)
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

    def showPolicyValues(self, states, objs):
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
                if self.obj and (i, j) in objs:
                    st = "----"

                out += str(st).ljust(6) + ' | '
            print(out)
        print('-------------------------------------------------------------------------------------------')
        print()

def get_plan(cost_map, new_initial_state, milestones_list, obj, lr, er, eps):
    if not milestones_list:
        return
    cost_map = list(map(int, cost_map))
    init = int(new_initial_state)
    dest = int(milestones_list[0])
    ag = Agent(start_state=int_to_pair(init), win_state=int_to_pair(dest),
               lr=lr, exp_rate=er, cm=cost_map, obj=obj)
    objs = ag.State.objs
    print("Start: ", datetime.datetime.now())  # Do not delete
    start_time = timer()
    ag.play(eps)
    print("State values computed using value iteration:")
    ag.showValues()
    end_time = timer()

    print("End: ", datetime.datetime.now())  # Do not delete
    print("Time taken               ", end_time - start_time)
    print("Policy for the above state values:")

    # The policy as an array of (state, action) pairs.
    s = ag.getPolicy()
    plan_coords = [c for c, a in s]
    cost = sum(cost_map[pair_to_int(i, j)] for i, j in plan_coords[1:])
    ag.showPolicyValues(s, objs)
    # *** Extract the actions from the policy. This will be used in integration into UI. ***
    return plan_coords, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path-planning using value iteration.")
    parser.add_argument("-o", "--obj",  default=False, action='store_true',
                        help="Whether using red blocks or not. Default: False.")
    parser.add_argument("-l", "--learning_rate", default=0.2, type=float, help="Learning rate")
    parser.add_argument("-e", "--exp_rate", default=0.5, type=float,
                        help="Exploration rate.")
    parser.add_argument("-eps", "--episodes", default=100, type=float,
                        help="Number of episodes to train for.")
    # TODO: Use this to toggle between the two agents.
    # parser.add_argument("-nd", "--nondet", default=False, action='store_true', help="Non-deterministic env or not")
    parser.add_argument("-cm", "--costmap", default=CM, nargs='+', help="Cost map")
    parser.add_argument("-i", "--init", default=MILESTONES[0], type=int, help="New initial state")
    parser.add_argument("-ms", "--milestones", default=MILESTONES[1:], nargs='+', help="List of remaining milestones")
    parser.add_argument("-s", "--single", default=False, action='store_true', help="Single path or iterate through all")

    # TODO: Implement timeout functionality.
    parser.add_argument("-to", "--timeout", default=2, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()

    if ARGS.single:
        if ARGS.init and (not ARGS.milestones or ARGS.init == ARGS.milestones[0]):
            raise Exception("List of milestones is required when initial state specified.")
        plan_coords, cost = get_plan(ARGS.costmap, ARGS.init, ARGS.milestones, ARGS.obj, ARGS.learning_rate, ARGS.exp_rate, ARGS.episodes)
        print("Plan: ", plan_coords)
        print("Total cost: ", cost)
        exit()

    start = 0
    end = 1
    total_length = 0
    p = []
    cost = 0
    init_plan_state = None
    total_plan = []
    cost_so_far = 0
    init_state = ARGS.init
    milestones_list = ARGS.milestones
    milestones = [init_state] + milestones_list
    while end < len(ARGS.milestones) + 1:
        init_state = milestones[start]
        milestones_list = milestones[end:]
        if not milestones_list:
            break
        # Repeatedly train rl agent to reach each milestone.
        plan_coords, cost = get_plan(ARGS.costmap, init_state, milestones_list, ARGS.obj, ARGS.learning_rate, ARGS.exp_rate, ARGS.episodes)
        cost_so_far += cost
        if init_plan_state is None:
            init_plan_state = plan_coords[0]
        total_plan += plan_coords[1:]
        print("Cost so far: ", cost_so_far)
        total_length += len(plan_coords) - 1
        print("Reached", NAMES.get(int(milestones[end]), "cell " + milestones[end]), "after", total_length, "blocks.")

        start += 1
        end += 1

    total_plan = [init_plan_state] + total_plan
    print("Total length: ", total_length, "blocks.")
    print("Plan: ", total_plan)
    print("Total cost: ", cost_so_far)

