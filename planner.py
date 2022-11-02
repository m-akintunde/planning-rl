import datetime
from timeit import default_timer as timer

from utils import int_to_pair, pair_to_int
from agents.qlagent import QLAgent
# from agents.viagent import VIAgent


class Planner:
    def __init__(self, args):
        self.nondet = args.nondet
        self.gamma = args.gamma
        self.show_qvals = args.show_qvals

    def get_plan(self, cost_map, new_initial_state, milestones_list, obj, lr, er, eps, p):
        cost_map = list(map(int, cost_map))
        init = int(new_initial_state)
        dest = int(milestones_list[0])

        if not milestones_list:
            return

        #if self.nondet:
        ag = QLAgent(start_state=int_to_pair(init), win_state=int_to_pair(dest),
                     lr=lr, exp_rate=er, cm=cost_map, obj=obj,
                     decay_gamma=self.gamma, prob=p, determine=not self.nondet)
        # else:
        #     ag = VIAgent(start_state=int_to_pair(init), win_state=int_to_pair(dest),
        #                  lr=lr, exp_rate=er, cm=cost_map, obj=obj)

        objs = ag.getObjs()
        emergency_objs = ag.getEmergencyObjs()
        print("Start: ", datetime.datetime.now())  # Do not delete
        start_time = timer()
        ag.play(eps)
        end_time = timer()
        print("End: ", datetime.datetime.now())  # Do not delete
        print("Time taken               ", end_time - start_time)
        print("State values computed using value iteration:")
        #ag.showValues()

        print("Policy:")

        # The policy as an array of (state, action) pairs.
        s = ag.getPolicy()
        ag.showPolicyValues(s, objs, emergency_objs)
        #ag.showValues()
        d = {}
        vs = ag.Q_values  # if self.nondet else ag.state_values
        for k, v in vs.items():
            d[pair_to_int(*k)] = v
            if self.show_qvals:
                print(pair_to_int(*k), v)

        plan_coords = [c for c, a in s]
        cost = sum(cost_map[pair_to_int(i, j)] for i, j in plan_coords[1:])

        # *** Extract the actions from the policy. This will be used in integration into UI. ***
        return plan_coords, cost, d

