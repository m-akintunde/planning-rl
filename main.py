import argparse

from planner import Planner
from utils import MILESTONES, CM, NAMES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path-planning using value iteration.")
    parser.add_argument("-o", "--obj",  default=False, action='store_true',
                        help="Whether using red blocks or not. Default: False.")
    parser.add_argument("-n", "--nondet",  default=False, action='store_true',
                        help="Whether operating in a non-deterministic env or not. Default: False.")
    parser.add_argument("-l", "--learning_rate", default=0.2, type=float, help="Learning rate")
    parser.add_argument("-e", "--exp_rate", default=0.5, type=float,
                        help="Exploration rate.")
    parser.add_argument("-eps", "--episodes", default=1000, type=float,
                        help="Number of episodes to train for.")
    parser.add_argument("-cm", "--costmap", default=CM, nargs='+', help="Cost map")
    parser.add_argument("-i", "--init", default=MILESTONES[0], type=int, help="New initial state")
    parser.add_argument("-ms", "--milestones", default=MILESTONES[1:], nargs='+', help="List of remaining milestones")
    parser.add_argument("-s", "--single", default=False, action='store_true',
                        help="Single path or iterate through all milestones")
    parser.add_argument("-q", "--show_qvals", default=False, action='store_true',
                        help="Whether to show q values or not. Default: False.")
    parser.add_argument("-g", "--gamma", default=0.9, type=float, help="Gamma decay rate")
    parser.add_argument("-p", "--prob", default=0.8, type=float,
                        help="Probability of successful transition. Otherwise perpendicular move with (1-p)/2 chance")
    parser.add_argument("-f", "--file", default="qvals.txt", type=str,
                        help="Path to store q values")
    parser.add_argument("-r", "--run_policy", default=False, action='store_true',
                        help="Run the generated policy.")

    # TODO: Implement timeout functionality.
    parser.add_argument("-to", "--timeout", default=2, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()
    planner = Planner(ARGS)
    pr = ARGS.prob if ARGS.nondet else 1
    if ARGS.single:
        if ARGS.init and (not ARGS.milestones or ARGS.init == ARGS.milestones[0]):
            raise Exception("List of milestones is required when initial state specified.")
        plan_coords, cost, vals = planner.get_plan(ARGS.costmap, ARGS.init, ARGS.milestones, ARGS.obj,
                                                   ARGS.learning_rate, ARGS.exp_rate, ARGS.episodes, pr)
        if ARGS.run_policy:
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
        plan_coords, cost, vals = planner.get_plan(ARGS.costmap, init_state, milestones_list, ARGS.obj,
                                                   ARGS.learning_rate, ARGS.exp_rate, ARGS.episodes, pr)
        if ARGS.run_policy:
            cost_so_far += cost
            if init_plan_state is None:
                init_plan_state = plan_coords[0]
            total_plan += plan_coords[1:]
            print("Cost so far: ", cost_so_far)
            total_length += len(plan_coords) - 1
            print("Reached", NAMES.get(int(milestones[end]), "cell " + milestones[end]), "after", total_length, "blocks.")

        start += 1
        end += 1

    if ARGS.run_policy:
        total_plan = [init_plan_state] + total_plan
        print("Total length: ", total_length, "blocks.")
        print("Plan: ", total_plan)
        print("Total cost: ", cost_so_far)

