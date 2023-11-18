import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from consts import state_variables

def quantiles_discretization(vector):

    new_vector = vector.copy(deep=True)

    # Change NaN values to 0 (no drug)
    new_vector = new_vector.fillna(0)

    # Calculate quantiles (only for the non-zero values!)
    non_zero_values = new_vector[new_vector > 0]
    q1 = non_zero_values.quantile(0.25)
    q2 = non_zero_values.quantile(0.50)
    q3 = non_zero_values.quantile(0.75)

    new_vector = new_vector.astype("object")

    new_vector.loc[vector == 0.0] = 0
    new_vector[(vector > 0.0) & (vector < q1)] = 1
    new_vector[(vector >=  q1) & (vector <  q2)] = 2
    new_vector[(vector >=  q2) & (vector < q3)] = 3
    new_vector[(vector >=  q3)] = 4

    return new_vector, [q1, q2, q3]


def clustering_states(data, state_variables, num_clusters):

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init='auto').fit(data[state_variables].to_numpy())

    return kmeans.labels_, kmeans.cluster_centers_


def add_states_and_rewards(data, state_clusters):
    """
    Returns the given dataframe, including state, next_state and reward columns
    In addition, drop patients with a trajectory of size 1
    """

    data_copy = data.copy(deep=True)

    # Add states to the dataframe
    data_copy["state"] = state_clusters

    # Drop patients that has only one row (meaning a patient with a trajectory of size 1)
    value_counts = data_copy['patientunitstayid'].value_counts()
    df_filtered = data_copy[data_copy['patientunitstayid'].map(value_counts) > 1].reset_index(drop=True)

    # Make sure the dataframw is sorted by patient and time
    df_filtered = df_filtered.sort_values(by=["patientunitstayid", "timestep_1h"]).reset_index(drop=True)

    # Add next state column
    # by group by 'patientunitstayid' and shift the 'state' column to get the previous state
    # note that the last row for each patient will be next_state=np.NaN, because there is no state after it
    df_filtered['next_state'] = df_filtered.groupby('patientunitstayid')['state'].shift(-1)

    # Add two terminal states (for the last row for each patient)
    num_states = df_filtered["state"].nunique()
    ALIVE_STATE = num_states
    EXPIRED_STATE = num_states + 1

    mask = df_filtered['next_state'].isna() # Create a mask for NaN values in 'next_state'
    df_filtered['next_state'] = np.where(mask & (df_filtered['unitdischargestatus'] == 'Alive'), ALIVE_STATE, df_filtered['next_state'])
    df_filtered['next_state'] = np.where(mask & (df_filtered['unitdischargestatus'] == 'Expired'), EXPIRED_STATE, df_filtered['next_state'])
    df_filtered['next_state'] = df_filtered['next_state'] .astype("int64")


    # Add reward column
    df_filtered["reward"] = np.zeros(df_filtered.shape[0])
    df_filtered['reward'] = np.where(mask & (df_filtered['unitdischargestatus'] == 'Alive'), 100, df_filtered['reward'])
    df_filtered['reward'] = np.where(mask & (df_filtered['unitdischargestatus'] == 'Expired'), -100, df_filtered['reward'])
    df_filtered['reward'] = df_filtered['reward'] .astype("int64")

    return df_filtered


def generate_test_set_states(test_set, state_centroids):
    """
    Input:      test_set: data frame of shape (n, features)

                state_centroids: array of shape (num_clusters, state_dim)


    Outputs:   test_set_states - a vector of length n. Includes discrete state assigment for each test sample
    """

    # Use pairwise_distances_argmin_min to find the closest cluster for each row in the test set
    test_set_states, _ = pairwise_distances_argmin_min(test_set[state_variables], state_centroids)

    return test_set_states



def create_transition_table(data, state_dim, action_dim):
    """

    Note:
    - Each action defines a 2D matrix T[a, :, :] which is describe the probability to move from each state to another by action a.
    - T[a, :, :] should be a stochastic matrix, but some transitions do not exisit in the data, so there are some rows which  contains only 0's (and hence doesnt sum up to 1).
    """

    T = np.zeros((action_dim, state_dim, state_dim))

    unique_state = data['state'].unique()
    unique_next_state = data['next_state'].unique()
    unique_action = data['action'].unique()

    # Group the DataFrame by action and calculate transition probabilities
    for action in unique_action:
        group = data[data['action'] == action]

        for state in unique_state:
            total_transitions = len(group[group['state'] == state])
            for next_state in unique_next_state:
                num_transitions = len(group[(group['state'] == state) & (group['next_state'] == next_state)])

                if total_transitions > 0:
                   T[action, state ,next_state] = num_transitions / total_transitions

    # Add probability 1 to move from terminal state to itself
    terminal_state_1 = state_dim-2
    terminal_state_2 = state_dim-1

    T[:,terminal_state_1,terminal_state_1] = 1
    T[:,terminal_state_2,terminal_state_2] = 1

    return T


def enviroment_generation(data, num_clusters):
    """
    Generate enviroment components.
    """

    df = data.copy(deep=True)

    # Discreteize Actions
    df["action_vaso"], vaso_quantiles = quantiles_discretization(data["drugrate_vaso"])
    df["action_iv_fluids"], iv_quantiles = quantiles_discretization(data["drugrate_iv_fluids"])

    # Add Action column
    # Create a column of a single action (using the vaso action and the iv_fluids action)
    action_combinations = [(i, j) for i in range(5) for j in range(5)]
    actions_map = {combo: index for index, combo in enumerate(action_combinations)}

    def map_actions_to_number(row):
        return actions_map[(row["action_vaso"], row["action_iv_fluids"])]

    df['action'] = df.apply(map_actions_to_number, axis=1)

    # Discreteize States, and create state, next_state, reward columns
    state_clusters, state_centroids = clustering_states(df, state_variables, num_clusters)
    df = add_states_and_rewards(df, state_clusters)

    # Create transition matrix
    state_dim = num_clusters + 2 # Clusters number + 2 terminal states
    action_dim = 25 # Actions are numbered from 0 to 24 (total: 25 possibole actions)
    T = create_transition_table(df, state_dim, action_dim)


    # Create reward vector
    ALIVE_STATE = num_clusters
    EXPIRED_STATE = num_clusters + 1
    R = np.zeros(state_dim)
    R[ALIVE_STATE] = 100
    R[EXPIRED_STATE] = -100

    return df, T, R, state_centroids, vaso_quantiles, iv_quantiles


