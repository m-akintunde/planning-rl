# Q-Learning agent
import argparse
from timeit import default_timer as timer
import datetime
import numpy as np

from main import State, MILESTONES
from utils import BOARD_ROWS, BOARD_COLS, CM, int_to_pair, pair_to_int, NAMES


class QLAgent:
    def __init__(self, start_state, win_state, cm, lr=0.2, exp_rate=0.5, decay_gamma=0.9, obj=True):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.win_state = win_state
        self.start_state = start_state
        self.obj = obj
        print("Constructing q-learning (non-deterministic) agent with start state", start_state, "win state", win_state)
        self.determine = False
        self.State = State(state=self.start_state, win_state=self.win_state, determine=self.determine, obj=self.obj, cm=cm)
        self.isEnd = self.State.isEnd
        self.decay_gamma = decay_gamma
        self.lr = lr
        self.cm = cm
        self.exp_rate = exp_rate

        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0
    def getPolicy(self):
        # "Reset" with initial state at the beginning of path.
        self.State = State(self.start_state, self.win_state, self.determine, self.cm, self.obj)
        self.states = [(self.State.state, "*")]
        while not self.State.isEnd:
            action = max(self.Q_values[self.State.state], key=self.Q_values[self.State.state].get)
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

    def showPolicyValues(self, objs, emergency_objs):
        directions = {"left": "<", "right": ">", "up": "^", "down": "v"}
        #for k, v in self.Q_values.items():
        #    print(k, v)
        for i in range(0, BOARD_ROWS):
            print('-------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                q_vals = self.Q_values[(i, j)]
                st = directions[max(q_vals, key=q_vals.get)]
                if (i, j) in emergency_objs or self.obj and (i, j) in objs:
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
            # If at the end of game back propagate reward
            if self.State.isEnd:
                # back propagate

                # Set all actions of the last state as the current reward (must be 1 since no fail state).
                # Helps to "converge faster".
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

def get_plan_nondet(cost_map, new_initial_state, milestones_list, obj, lr, er, eps, gamma, q):
    if not milestones_list:
        return
    cost_map = list(map(int, cost_map))
    init = int(new_initial_state)
    dest = int(milestones_list[0])
    ag = QLAgent(start_state=int_to_pair(init), win_state=int_to_pair(dest),
                 lr=lr, exp_rate=er, decay_gamma=gamma,
                 obj=obj, cm=cost_map)

    objs = ag.State.objs
    emergency_objs = ag.State.emergency_objs
    print("Start: ", datetime.datetime.now())  # Do not delete
    start_time = timer()
    ag.play(eps)
    #print("State values computed using value iteration:")

    end_time = timer()
    # print("Latest Q values computed using q learning:")
    # print(ag.Q_values)
    print("End: ", datetime.datetime.now())  # Do not delete
    print("Time taken               ", end_time - start_time)
    print("Policy for the computed q values:")

    # The policy as an array of (state, action) pairs.
    s = ag.getPolicy()
    ag.showPolicyValues(objs, emergency_objs)
    d = {}
    for k, v in ag.Q_values.items():
        d[pair_to_int(*k)] = v
        if q:
            print(pair_to_int(*k), v)
    plan_coords = [c for c, a in s]
    cost = sum(cost_map[pair_to_int(i, j)] for i, j in plan_coords[1:])
    # *** Extract the actions from the policy. This will be used in integration into UI. ***

    return plan_coords, cost, d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path-planning using q-learning.")
    parser.add_argument("-o", "--obj", default=False, action='store_true',
                        help="Whether using red blocks or not. Default: False.")

    parser.add_argument("-l", "--learning_rate", default=0.2, type=float, help="Learning rate")
    parser.add_argument("-g", "--gamma", default=0.9, type=float, help="Gamma decay rate")
    parser.add_argument("-e", "--exp_rate", default=0.5, type=float,
                        help="Exploration rate.")
    parser.add_argument("-eps", "--episodes", default=100, type=float,
                        help="Number of episodes to train for.")
    parser.add_argument("-cm", "--costmap", default=CM, nargs='+', help="Cost map")
    parser.add_argument("-i", "--init", default=MILESTONES[0], type=int, help="New initial state")
    parser.add_argument("-ms", "--milestones", default=MILESTONES[1:], nargs='+', help="List of remaining milestones")
    parser.add_argument("-s", "--single", default=False, action='store_true',
                        help="Single path or iterate through all milestones")
    parser.add_argument("-q", "--show_qvals", default=False, action='store_true',
                        help="Whether to show q values or not. Default: False.")
    # TODO: Implement timeout functionality.
    parser.add_argument("-to", "--timeout", default=2, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()
    init_plan_state = None
    start = 0
    end = 1
    total_length = 0
    obj = ARGS.obj
    p = []
    cost_so_far = 0
    total_plan = []

    init_state = ARGS.init
    milestones_list = ARGS.milestones
    milestones = [init_state] + milestones_list

    if ARGS.single:
        if ARGS.init and (not ARGS.milestones or ARGS.init == ARGS.milestones[0]):
            raise Exception("List of milestones is required when initial state specified.")
        plan_coords, cost = get_plan_nondet(ARGS.costmap, init_state, milestones_list, ARGS.obj, ARGS.learning_rate,
                                            ARGS.exp_rate, ARGS.episodes, ARGS.gamma, ARGS.show_qvals)
        print("Plan: ", plan_coords)
        print("Total cost: ", cost)
        exit()

    while end < len(ARGS.milestones) + 1:
        init_state = milestones[start]
        milestones_list = milestones[end:]
        if not milestones_list:
            break
        # Repeatedly train rl agent to reach each milestone.
        plan_coords, cost, q_vals = get_plan_nondet(ARGS.costmap, init_state, milestones_list, ARGS.obj, ARGS.learning_rate,
                                            ARGS.exp_rate, ARGS.episodes, ARGS.gamma, ARGS.show_qvals)
        cost_so_far += cost
        #total_plan.append(plan_coords[0])
        if init_plan_state is None:
            init_plan_state = plan_coords[0]
        total_plan += plan_coords[1:]
        print("Cost so far: ", cost_so_far)
        total_length += len(plan_coords) - 1
        print("Reached", NAMES.get(int(milestones[end]), "cell " + str(milestones[end])), "after", total_length, "blocks.")
        start += 1
        end += 1

    total_plan = [init_plan_state] + total_plan
    print("Total length: ", total_length, "blocks.")
    print("Plan: ", total_plan)
    print("Total cost: ", cost_so_far)