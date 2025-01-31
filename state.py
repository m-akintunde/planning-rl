import numpy as np

from utils import BOARD_ROWS, BOARD_COLS, int_to_pair, NOT_EMERGENCY_COSTS, pair_to_int


# Representation of the gridworld.
class State:
    def __init__(self, state, win_state, determine, cm, obj=True, prob=0.8):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.obj = obj
        self.cm = cm
        self.p = prob

        # TODO: Are these ever used?
        self.emergency_objs = [int_to_pair(i) for i, j in enumerate(cm) if j not in NOT_EMERGENCY_COSTS]
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
        cm_value = self.cm[pair_to_int(*self.state)]
        if self.state == self.win_state:
            return 1
        elif cm_value == 1:
            return -1
        elif cm_value not in (1, 3):
            return -10 * cm_value
        elif cm_value == 3:
            return -3 if self.obj else -1

    def isEndFunc(self):
        if self.state == self.win_state:
            self.isEnd = True

    def _chooseActionProb(self, action):
        probs = [self.p, (1-self.p)/2, (1-self.p)/2]
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=probs)
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=probs)
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=probs)
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=probs)

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
                return nxtState
        return self.state
