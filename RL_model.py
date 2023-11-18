import numpy as np


def create_AI_policy(T, R, action_dim, state_dim, gamma, epsilon):
    """
    @T: Transition matrix. a numpy array of shape (action_dim x state_dim+2 x state_dim+2)
    @R: Reward vector. a numpy array of shape (state_dim+2,)
    @action_dim:
    @state_dim:
    @gamma: reward decay rate. Different values of gamma may produce different policies.
            Lower gamma values will put more weight on short-term gains,
            whereas higher gamma values will put more weight towards long-term gains.
    #epsilon: The allowable norm of the difference between the old policy and the new policy, so we know when the optimal value has been reached.


    return: policy:

            an array of shape (K,).  where K=state_dim+2.
            This is the deterministic optimal policy - AI's recommended action to take for each state.

            value_vector:

    """

    # Initilize the policy vector to random numbers, except for the terminal states, which have a policy of 0.
    K = len(R) # K is state_dim+2
    policy_vector = np.random.randint(0, action_dim, (K, ), dtype="int64") # policy_vector is of shape (K,)
    ALIVE_STATE = state_dim - 1
    EXPIRED_STATE = state_dim - 2

    policy_vector[ALIVE_STATE] = 0
    policy_vector[EXPIRED_STATE] = 0
    new_policy_vector = np.zeros(K, dtype="int64")


    # Initialize the value vector
    value_vector = np.zeros(K) # value_vector is of shape (K,)
    value_vector_old = value_vector

    # Set any NaN values in the transition matrix to zero (there shouldn't be any, but we're being safe)
    # T(isnan(T)) = 0;

    iteration = 0

    # This while loop is where Policy Iteration is performed.  For more information on Policy Iteration, see here:
    # https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
    while np.all(new_policy_vector == policy_vector) == False: # np.linalg.norm(new_policy_vector - policy_vector) > epsilon:

        if iteration > 1:
           policy_vector = new_policy_vector

        # Policy Evaluation
        for s in range(K-2):
            a = policy_vector[s]

            # sum_term = np.sum(T[a,s,:] * value_vector_old)
            sum_term = 0
            for s_prime in range(K):
                sum_term = sum_term + T[a,s,s_prime] * value_vector_old[s_prime] # T[a][s][s_prime]is the probability to move from state s to s_prime by a action

            expected_reward = np.sum(T[a,s,:] * R)
            value_vector[s] = expected_reward + gamma * sum_term

        value_vector_old = value_vector

        # Policy Improvment
        for s in range(K-2):
            best_action = policy_vector[s]
            best_value  = min(value_vector_old)

            for a in range(action_dim):

                sum_term = 0
                for s_prime in range(K-2):
                    sum_term = sum_term + T[a,s,s_prime] * value_vector_old[s_prime]

                expected_reward = np.sum(T[a,s,:] * R)
                value = expected_reward + gamma * sum_term

                if value > best_value:
                    best_value  = value
                    best_action = a

            new_policy_vector[s] = best_action

        iteration += 1

    policy = policy_vector

    return policy, value_vector