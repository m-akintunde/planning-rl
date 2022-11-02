from state import State
from utils import pair_to_int


class Agent:
    def __init__(self, start_state, win_state, cm, lr=0.2, exp_rate=0.5, obj=True):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.win_state = win_state
        self.start_state = start_state
        self.obj = obj
        self.determine = False
        self.State = State(state=self.start_state, win_state=self.win_state, determine=self.determine,
                           obj=self.obj, cm=cm)
        self.lr = lr
        self.cm = cm

        # Probability of choosing a random action (exploration) rather than choosing an action
        # greedily based on what has already been learned.
        self.exp_rate = exp_rate

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
            print("cost", self.cm[pair_to_int(*self.State.state)])
            print("---------------------")
        return self.states

    def chooseAction(self):
        pass

    def showPolicyValues(self, states, objs, emergency_objs):
        pass

    def play(self, rounds=10):
        pass

    def reset(self):
        pass

    def takeAction(self, action):
        pass

    def chooseNondetPolicyAction(self):
        pass

    def getNondetPolicy(self):
        pass

    def choosePolicyAction(self):
        pass

    def getObjs(self):
        return self.State.objs

    def getEmergencyObjs(self):
        return self.State.emergency_objs
