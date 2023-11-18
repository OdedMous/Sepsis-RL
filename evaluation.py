import random
import numpy as np


def create_clinician_policy(states, actions, state_dim, action_dim):
    """
    Inputs:     states:   Vector of the discrete patient states.

                actions:  Vector of the real clinical actions (after
                          discretization) corresponding to the discrete patient
                          states.


     Outputs:  clinician_policy:  K x |A| matrix (where |A| is the number of
                                   actions) giving the probability that given
                                   state i, the clinician will choose action j.

      This function takes the vectors of states and their corresponding
      clinician actions and creates the clinician dosing policy by empirically
      counting the number of times a clinician used a given action for a given
      state in the cohort.
    """

    clinician_policy = np.zeros((state_dim, action_dim))

    # Count the instances of doing action j in state i.
    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        clinician_policy[state][action] += 1

    # Convert empirical counts to observed probabilities.
    for i in range(state_dim):

        normalization = np.sum(clinician_policy[i, :])

        if normalization != 0:
           clinician_policy[i, :] /= normalization


    return clinician_policy


def compute_next_state(T, current_state, action):
    """
    Given current_state and action, sample the next state using the transition matrix T.
    """

    state_dim = T.shape[1]

    # Get the probability vector for the given current_state and action
    probability_vector = T[action, current_state, :]

    if sum(probability_vector) == 0: # if the the probability vector is  0's vector, than make it uniform distribution
       probability_vector = np.ones(state_dim) / state_dim

    # Create an array of possible states from 0 to num_states-1
    possible_states = np.arange(state_dim)

    # Sample the next state based on the probability vector
    next_state = np.random.choice(possible_states, p=probability_vector)

    return next_state


def create_trials(clinician_policy, test_set_states, R, T, num_trials, max_steps, num_actions):

    # Initialize the set of trials that will store all trial data.
    trials = [{'trial_number': i,
               'clinician_steps': None,
               'clinician_states': [0] * max_steps,
               'clinician_actions': [0] * max_steps,
               'clinician_rewards': [0] * max_steps,
               'rho': [0] * max_steps}
               for i in range(num_trials)]


    # Fill the trials object
    for i in range(num_trials):
        initial_state = random.choice(test_set_states)
        state = initial_state
        trial_reward = 0
        num_steps = 1

        while trial_reward == 0 and num_steps < max_steps:
              trials[i]["clinician_states"][num_steps] = state

              # Given a state, select a random action based on the clinican_policy probabilities in this state
              action = random.choices(range(num_actions), clinician_policy[state], k=1)[0]
              trials[i]["clinician_actions"][num_steps] = action

              state = compute_next_state(T, state, action)
              trial_reward = R[state]
              trials[i]["clinician_rewards"][num_steps] = trial_reward

              num_steps += 1

        trials[i]["clinician_steps"] = num_steps

    return trials

def compute_WIS(trials, clinician_policy, ai_policy, num_trials, max_steps, num_actions, K, gamma, policy_type):
    """
    """

    clinican_action_freq = np.zeros(num_actions)
    ai_action_freq = np.zeros(num_actions)


    # Compute rho
    # pi_clinician - The probability P(a|s) according to the observed train data
    # pi_ai - 0.99 ,      if a_ai == a_clinican,
    #         0.01 / K ,  else
    for i in range(num_trials):

        num_steps = trials[i]["clinician_steps"]

        for j in range(num_steps):
            state = trials[i]["clinician_states"][j]
            action = trials[i]["clinician_actions"][j]

            # Define ai_action based on the ai policy type
            if policy_type == "AI_RL":
               ai_action = ai_policy[state]
            if policy_type == "ZERO_DRUG":
               ai_action = 0
            if policy_type == "RANDOM":
               ai_action = random.randint(0, num_actions-1)

            clinican_action_freq[action] += 1 / (num_steps * num_trials)
            ai_action_freq[ai_action] += 1 / (num_steps * num_trials) #

            if ai_action == action:
                pi_ai = 0.99
            else:
                pi_ai = 0.01 / K

            pi_clinician = clinician_policy[state, action]
            rho = pi_ai / pi_clinician if pi_clinician != 0 else 0 #########################
            trials[i]["rho"][j] = rho


        # Assign 1 to the remaining elements in trial[i]["rho"]
        trials[i]["rho"][num_steps:] = [1] * (len(trials[i]["rho"]) - num_steps)

    # Compute W for each time point.
    w = np.zeros(max_steps)

    for i in range(max_steps):
        for j in range(num_trials):
            w[i] += np.prod(trials[j]['rho'][:i])

    w = w * (1 / num_trials)

    # Compute the value for the clinician and the AI Clinician.
    gamma_vector = np.power(gamma, np.arange(max_steps + 1))
    V_vector = np.zeros(num_trials)
    V_vector_clinician = np.zeros(num_trials)

    for i in range(num_trials):
        num_steps = trials[i]['clinician_steps']
        rho = np.prod(trials[i]['rho'][:num_steps])
        reward = trials[i]['clinician_rewards'][:num_steps]
        w_H = w[num_steps]

        V_vector[i] = (rho / w_H) * np.sum(gamma_vector[:num_steps] * reward) if w_H !=0 else 0 ##############
        V_vector_clinician[i] = np.sum(gamma_vector[:num_steps] * reward)

    V_WIS = np.mean(V_vector)
    V_clinician = np.mean(V_vector_clinician)

    return V_WIS, V_clinician, clinican_action_freq, ai_action_freq

def evaluation(ai_policy, clinician_policy, test_set_states, T, gamma, R):


    num_trials = 200 # 10000
    max_steps = 1000
    K = clinician_policy.shape[0]
    num_actions = clinician_policy.shape[1]



    result = {"AI_RL":     {"V_WIS": None, "V_clinician": None, "clinican_action_freq": None, "ai_action_freq": None},
              "ZERO_DRUG": {"V_WIS": None, "V_clinician": None, "clinican_action_freq": None, "ai_action_freq": None},
              "RANDOM":    {"V_WIS": None, "V_clinician": None, "clinican_action_freq": None, "ai_action_freq": None}}


    for policy_type in ["AI_RL", "ZERO_DRUG", "RANDOM"]:

        trials = create_trials(clinician_policy, test_set_states, R, T, num_trials, max_steps, num_actions)           # Initialize the set of trials that will store all trial data.

        V_WIS, V_clinician, clinican_action_freq, ai_action_freq = compute_WIS(trials, clinician_policy, ai_policy, num_trials, max_steps, num_actions, K, gamma, policy_type)

        result[policy_type]["V_WIS"] = V_WIS
        result[policy_type]["V_clinician"] = V_clinician
        result[policy_type]["clinican_action_freq"] = clinican_action_freq
        result[policy_type]["ai_action_freq"] = ai_action_freq



    return result


