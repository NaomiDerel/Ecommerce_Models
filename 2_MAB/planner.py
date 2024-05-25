import numpy as np
import pandas as pd
import time

TIME_CAP = 2 * (10 ** 2)


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        # Save all the given parameters:
        self.num_rounds = num_rounds
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_users = num_users
        self.arms_thresh = arms_thresh
        self.users_distribution = users_distribution

        # Update information:
        self.last_arm_sampled = -1
        self.last_user_sampled = -1
        self.current_round = 0
        self.total_rounds = 0
        self.exposure_list = np.zeros(self.num_arms)
        self.arm_status = np.ones(num_arms)  # 1=activated, 0=deactivated

        # UCB Information:
        self.uniform_mle = np.zeros((num_arms, num_users))  # maximum likelihood estimator
        self.arms_to_keep = [arm for arm in range(num_arms)]  # start from all arms
        self.total_samples = np.zeros((num_arms, num_users))  # n(a, u)

        # User preferences for arms:
        # data = {}
        # for i in range(num_users):
        #     data['user ' + str(i)] = range(num_arms)
        # self.preferences = pd.DataFrame(data)

        # Simulations hyper-parameters:
        K = num_arms * num_users
        self.end_of_explore = int(((num_rounds / K) ** (2 / 3)) * (np.log(num_rounds) ** (1 / 3)))

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        # Find number of rounds needed to not lose any arm we want to keep:
        needed_rounds = sum(max(0, self.arms_thresh[arm] - self.exposure_list[arm]) for arm in self.arms_to_keep)
        chosen_arm = -1

        # If just enough rounds are left, interject to not lose any arms:
        if self.current_round >= (self.phase_len - needed_rounds):
            # min_score = np.inf
            max_score = -1
            # user preference:
            # user_row = self.preferences["user " + str(user_context)].tolist()

            for arm in self.arms_to_keep:
                exposure_diff = self.arms_thresh[arm] - self.exposure_list[arm]  # needs to meet threshold
                if exposure_diff > 0:
                    # out of arms that need to meet the threshold, pick the best one for the user:
                    # score = user_row.index(arm)
                    # if score < min_score:
                    #     chosen_arm = arm

                    score = self.uniform_mle[arm][user_context]
                    if score > max_score:
                        chosen_arm = arm

        # If an arm wasn't chosen, choose an arm with UCB:
        if chosen_arm == -1:
            sampled_rewards = np.zeros(self.num_arms)
            for arm in range(self.num_arms):
                # Don't choose inactive arms:
                if self.arm_status[arm] == 0:
                    sampled_rewards[arm] = 0
                # If the combination of user and arm was never chosen before, choose it:
                elif self.total_samples[arm][user_context] == 0:
                    chosen_arm = arm
                    break
                # Otherwise, calculate UCB from user x arm distribution and add Rad, then sample it:
                else:
                    UCB = self.uniform_mle[arm, user_context] + \
                           np.sqrt((2 * np.log(self.num_rounds)) / self.total_samples[arm][user_context])
                    sampled_rewards[arm] = np.random.uniform(0, UCB)

            # If an arm wasn't picked for the first time, choose the best sample:
            if chosen_arm == -1:
                chosen_arm = np.argmax(sampled_rewards)

        # Update data we need for the outcome calculations:
        self.current_round += 1
        self.total_rounds += 1
        self.exposure_list[chosen_arm] += 1
        self.total_samples[chosen_arm][user_context] += 1
        self.last_arm_sampled = chosen_arm
        self.last_user_sampled = user_context

        # Suggest the best arm found:
        return chosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # At the end of each phase, restart counters and update deactivated arms:
        if self.current_round % self.phase_len == 0:  # satisfied only when it is the end of the phase
            for arm in range(self.num_arms):
                if self.exposure_list[arm] < self.arms_thresh[arm]:
                    self.arm_status[arm] = 0
            self.current_round = 0
            self.exposure_list = np.zeros(self.num_arms)

            # Update user preferences by sorting the estimator for each arm:
            # for user in range(self.num_users):
            #     pref = {}
            #     for arm in range(self.num_arms):
            #         pref[arm] = self.uniform_mle[arm, user]
            #     sorted_pref = sorted(pref.items(), key=lambda x: x[1], reverse=True)
            #     self.preferences["user " + str(user)] = [x[0] for x in sorted_pref]

        # UCB distribution updates:
        user = self.last_user_sampled
        arm = self.last_arm_sampled
        self.uniform_mle[arm, user] = max(self.uniform_mle[arm, user], reward)
        # self.uniform_me[arm, user] = ((t - 1) / t) * self.uniform_me[arm, user] + (1 / t) * normalized_reward

        # If the end of the Explore Phase was reached, activate simulations:
        if self.total_rounds == self.end_of_explore:
            # All the possible combinations of arms:
            arm_groups = find_all_arm_groups(self.num_arms)
            max_reward = 0
            best_group = list(range(self.num_arms))  # current group

            # Define length of simulations according to number of groups:
            num_tests = int(self.num_rounds / (5 * len(arm_groups)))
            time_limit = (0.5*TIME_CAP / (self.num_rounds + len(arm_groups)*num_tests)) * num_tests
            print(f"num tests: {num_tests}, time limit: {time_limit}")

            # Create a simulation for each group:
            for group in arm_groups:
                total_reward = mini_simulation(num_tests, self.phase_len, self.num_arms, group,
                                               self.num_users, self.arms_thresh, self.users_distribution,
                                               self.uniform_mle, time_limit)
                print(f"{group}: {total_reward}")
                # Update the best group found:
                if total_reward > max_reward:
                    best_group = group
                    max_reward = total_reward

            # Update arms_to_keep to the best group found:
            for arm in range(self.num_arms):
                if arm not in best_group:
                    self.arms_to_keep.remove(arm)

    def get_id(self):
        return "id_codes"


class MiniPlanner(Planner):
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        super().__init__(num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution)

    def update_active_arms(self, arms):
        """
        :input: group of arms we want to keep in this planner.
        """
        self.arms_to_keep = arms

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # At the end of each phase, restart counters and update deactivated arms:
        if self.current_round % self.phase_len == 0:  # satisfied only when it is the end of the phase
            for arm in range(self.num_arms):
                if self.exposure_list[arm] < self.arms_thresh[arm]:
                    self.arm_status[arm] = 0
            self.current_round = 0
            self.exposure_list = np.zeros(self.num_arms)

            # Update user preferences by sorting the estimator for each arm:
            # for user in range(self.num_users):
            #     pref = {}
            #     for arm in range(self.num_arms):
            #         pref[arm] = self.uniform_mle[arm, user]
            #     sorted_pref = sorted(pref.items(), key=lambda x: x[1], reverse=True)
            #     self.preferences["user " + str(user)] = [x[0] for x in sorted_pref]

        # UCB distribution updates:
        user = self.last_user_sampled
        arm = self.last_arm_sampled
        self.uniform_mle[arm, user] = max(self.uniform_mle[arm, user], reward)


def find_all_arm_groups(num_arms):
    """
    :input: number of arms.
    :output: a list with all possible groups of arms with length >= 1, in total 2^num_arms - 1.
    """
    # recursive function:
    def generate_groups(arms, group_size, start_index, current_group, all_groups):
        if group_size == 0:
            all_groups.append(current_group.copy())
            return
        for i in range(start_index, len(arms)):
            current_group.append(arms[i])
            generate_groups(arms, group_size - 1, i + 1, current_group, all_groups)
            current_group.pop()

    # find all groups:
    arms = list(range(num_arms))
    groups = []
    for group_size in range(1, num_arms + 1):
        generate_groups(arms, group_size, 0, [], groups)
    return groups


def mini_simulation(num_rounds, phase_len, num_arms, arms, num_users, arms_thresh, users_distribution, estimated_ERM,
                    time_limit):
    """
    :input: the instance parameters similar to planner, and additionally:
        arms - group of active arms for this simulation round
        estimated_ERM - the distribution we estimated in the Planner, as the rewards function.
        time_limit - signal to stop the simulation sooner than expected because of the time limit.
    :output: normalized reward for the group tested.
    """
    # init mini planner:
    mini_planner = MiniPlanner(num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution)
    mini_planner.update_active_arms(arms)

    # outer scope parameters:
    total_reward = 0
    total_rounds_ran = 0
    begin_time = time.time()

    # test simulation:
    for i in range(num_rounds):
        # sample user:
        user_context = int(np.random.choice(range(num_users), size=1, p=users_distribution))
        # let mini planner choose best arm:
        chosen_arm = mini_planner.choose_arm(user_context)
        # get reward for round assuming legal choices and estimated distribution:
        reward = np.random.uniform(0, estimated_ERM[chosen_arm][user_context])
        # update mini planner:
        mini_planner.notify_outcome(reward)
        # update parameters:
        total_reward += reward
        total_rounds_ran += 1

        # check the calculations are not too slow:
        if time.time() - begin_time > time_limit:
            print("ran out of time")
            break

    # return the reward normalized by number of rounds preformed before running out of time:
    return total_reward / total_rounds_ran
