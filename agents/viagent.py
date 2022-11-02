# # Value-iteration agent
# import numpy as np
#
# from agents.agent import Agent
# from state import State
# from utils import BOARD_ROWS, BOARD_COLS
#
#
# class VIAgent(Agent):
#
#     def __init__(self, start_state, win_state, cm, lr=0.3, exp_rate=0.5, obj=True):
#         print("Constructing value iteration (deterministic) agent with start", start_state, "win state", win_state)
#         super().__init__(start_state, win_state, cm, lr, exp_rate, obj)
#         # initial state reward
#         self.state_values = {}
#         for i in range(BOARD_ROWS):
#             for j in range(BOARD_COLS):
#                 self.state_values[(i, j)] = 0  # set initial value to 0
#
#     def chooseAction(self):
#         # choose action with most expected value
#         mx_nxt_reward = 0
#         action = ""
#
#         if np.random.uniform(0, 1) <= self.exp_rate or \
#                 all(r == 0 for r in (self.State.nxtPosition(a) for a in self.actions)):
#             action = np.random.choice(self.actions)
#         else:
#             # greedy action, but only if there are some non-zero rewards to greedily choose from.
#             for a in self.actions:
#                 # if the action is deterministic
#                 nxt_reward = self.state_values[self.State.nxtPosition(a)]
#                 if nxt_reward >= mx_nxt_reward:
#                     action = a
#                     mx_nxt_reward = nxt_reward
#         return action
#
#     def choosePolicyAction(self):
#         mx_nxt_reward = 0
#         action = ""
#         for a in self.actions:
#             # if the action is deterministic
#             nxt_pos = self.State.nxtPolicyPosition(a)
#             if nxt_pos == self.State.state or (len(self.states) > 0 and nxt_pos in [s[0] for s in self.states]):
#                 nxt_reward = 0
#             else:
#                 nxt_reward = self.state_values[nxt_pos]
#             if nxt_reward > mx_nxt_reward:
#                 action = a
#                 mx_nxt_reward = nxt_reward
#         return action
#
#     def showPolicyValues(self, states, objs, emergency_objs):
#         directions = {"left": "<", "right": ">", "up": "^", "down": "v"}
#         for i in range(0, BOARD_ROWS):
#             print('-------------------------------------------------------------------------------------------')
#             out = '| '
#             for j in range(0, BOARD_COLS):
#                 st = self.state_values[(i, j)]
#                 sts = [s for s, _ in states]
#                 sts_map = dict(states)
#                 if (i, j) in sts:
#                     st = directions.get(sts_map[(i, j)], "*")
#                 if (i, j) in emergency_objs or self.obj and (i, j) in objs:
#                     st = "----"
#
#                 out += str(st).ljust(6) + ' | '
#             print(out)
#         print('-------------------------------------------------------------------------------------------')
#         print()
#
#     def play(self, rounds=10):
#         i = 0
#         while i < rounds:
#             # to the end of game back propagate reward
#             if self.State.isEnd:
#                 # back propagate
#                 reward = self.State.giveReward()
#                 # explicitly assign end state to reward values
#                 self.state_values[self.State.state] = reward  # this is optional
#                 #print("Game End Reward", reward)
#                 for s in reversed(self.states):
#                     reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
#                     self.state_values[s] = round(reward, 3)
#                 self.reset()
#                 i += 1  # End of episode, increase counter until reached max number of episodes.
#             else:
#                 action = self.chooseAction()
#                 # append trace
#                 self.states.append(self.State.nxtPosition(action))
#                 #print("current position {} action {}".format(self.State.state, action))
#                 # by taking the action, it reaches the next state
#                 self.State = self.takeAction(action)
#                 # mark is end
#                 self.State.isEndFunc()
#                 #print("nxt state", self.State.state)
#                 #print("---------------------")
#                 # If at the end of game back propagate reward
#             if self.State.isEnd:
#                 # back propagate
#
#                 # Set all actions of the last state as the current reward (must be 1 since no fail state).
#                 # Helps to "converge faster".
#                 reward = self.State.giveReward()
#                 self.state_values[self.State.state] = reward
#                 # print("Game End Reward", reward)
#                 self.reset()
#                 i += 1  # End of episode, increase counter until reached max number of episodes.
#                 # print("Episode:", i)
#             else:
#                 action = self.chooseAction()
#                 # append trace
#                 self.states.append(self.State.nxtPosition(action))
#                 # old state
#                 s = self.State.state
#
#                 # print("current position {} action {}".format(self.State.state, action))
#                 # by taking the action, it reaches the next state
#                 self.State = self.takeAction(action)
#                 r = self.State.giveReward()
#
#                 best_next_action = max(self.Q_values[self.State.state], key=self.Q_values[self.State.state].get)
#                 td_target = r + self.decay_gamma * self.Q_values[self.State.state][best_next_action]
#                 td_delta = td_target - self.Q_values[s][action]
#                 self.state_values[s][action] += self.lr * td_delta
#
#                 # mark is end
#                 self.State.isEndFunc()
#                 # print("nxt state", self.State.state)
#                 # print("---------------------")
#                 self.isEnd = self.State.isEnd
#
#     def reset(self):
#         self.states = []
#         self.State = State(self.start_state, self.win_state, True, self.cm, self.obj)
#
#     def takeAction(self, action):
#         position = self.State.nxtPosition(action)
#         return State(position, self.win_state, True, self.cm, self.obj)
#
#     def showValues(self):
#         for i in range(0, BOARD_ROWS):
#             print('-------------------------------------------------------------------------------------------')
#             out = '| '
#             for j in range(0, BOARD_COLS):
#                 s = self.state_values[(i, j)]
#                 out += str(s).ljust(6) + ' | '
#             print(out)
#         print('----------------------------------')
#
#     def showValuesStr(self):
#         o = ''
#         for i in range(0, BOARD_ROWS):
#             o += '\n-------------------------------------------------------------------------------------------'
#             out = '| '
#             for j in range(0, BOARD_COLS):
#                 s = self.state_values[(i, j)]
#
#                 out += str(s).ljust(6) + ' | '
#             o += '\n' + out
#         o += '\n-------------------------------------------------------------------------------------------'
#         return o
#
