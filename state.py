import numpy as np

from utils import BOARD_ROWS, BOARD_COLS, int_to_pair, EMERGENCY_COSTS

# Representation of the gridworld.
class State:
    def __init__(self, state, win_state, determine, cm, obj=True):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.obj = obj
        self.cm = cm

        self.emergency_objs = [int_to_pair(i) for i, j in enumerate(cm) if j in EMERGENCY_COSTS]
        self.objs = [int_to_pair(i) for i, j in enumerate(cm) if j == 3] + self.emergency_objs
        for o in self.emergency_objs:
            self.board[o[0], o[1]] = -1
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
                    if nxtState not in self.emergency_objs:
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
                    if nxtState not in self.emergency_objs:
                        return nxtState
                # TODO: Treat red blocks as states with negative reward rather than pure obstacle.
                elif self.obj and nxtState not in self.objs:
                    return nxtState
        return self.state
