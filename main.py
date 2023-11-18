import random
from tqdm import tqdm
import sys
import pandas as pd

from RL_model import create_AI_policy
from enviroment_generation import enviroment_generation, generate_test_set_states
from evaluation import create_clinician_policy, evaluation
from visualizations import plot_estimated_policy_value


def generate_train_test_data_indices(indices_list, training_proportion):

    # Shuffle the input list randomly
    random.shuffle(indices_list)

    # Calculate the split index based on the training proportion
    total_samples = len(indices_list)
    train_split = int(total_samples * training_proportion)

    # Split the list into training and testing sets
    training_indices = indices_list[:train_split]
    testing_indices = indices_list[train_split:]

    return training_indices, testing_indices


def generate_train_validation_test_data_indices(indices_list, training_proportion, validation_proportion):
    # Shuffle the input list randomly
    random.shuffle(indices_list)

    # Calculate the split indices based on proportions
    total_samples = len(indices_list)
    train_split = int(total_samples * training_proportion)
    val_split = int(total_samples * (training_proportion + validation_proportion))

    # Split the list into training, validation, and testing sets
    training_indices = indices_list[:train_split]
    validation_indices = indices_list[train_split:val_split]
    testing_indices = indices_list[val_split:]

    return training_indices, validation_indices, testing_indices



def main(data):

    # General parameters
    training_proportion = 0.8
    # validation_proportion = 0.2
    num_models = 100  # 100 # 500  realizations of the environment

    # Enviroment parameters
    action_dim = 25
    num_clusters = 18
    state_dim = num_clusters + 2
    ALIVE_STATE = num_clusters
    EXPIRED_STATE = num_clusters + 1

    # RL model parametrs
    gamma = 0.99
    epsilon = 0.0001

    # Variables
    V_WIS_lst = []  # AI policy
    V_clinician_lst = []  # clinicans policy
    V_WIS_zero_drug_lst = []  # zero drug policy
    V_WIS_random_lst = []  # random policy

    best_policy = {"value_func": None, "train_df": None, "T": None, "V_WIS": 0, "test_indices": None,
                   "test_states": None, "clinican_action_freq": None, "ai_action_freq": None,
                   'vaso_quantiles': None, 'iv_quantiles': None}

    # Calculation
    for i in tqdm(range(num_models)):

        indices_list = list(data.index)
        # train_indices, val_indices, test_indices = generate_train_validation_test_data_indices(indices_list, training_proportion, validation_proportion)
        train_indices, test_indices = generate_train_test_data_indices(indices_list, training_proportion)

        train_df, T, R, state_centroids, vaso_quantiles, iv_quantiles = enviroment_generation(data.iloc[train_indices],
                                                                                              num_clusters)

        ai_policy, value_func = create_AI_policy(T, R, action_dim, state_dim, gamma, epsilon)
        clinician_policy = create_clinician_policy(train_df["state"], train_df["action"], state_dim, action_dim)

        test_states = generate_test_set_states(data.iloc[test_indices], state_centroids)

        result = evaluation(ai_policy, clinician_policy, test_states, T, gamma, R)

        V_WIS, V_clinician, clinican_action_freq, ai_action_freq = result["AI_RL"]["V_WIS"], result["AI_RL"][
            "V_clinician"], result["AI_RL"]["clinican_action_freq"], result["AI_RL"]["ai_action_freq"]
        V_WIS_zero_drug = result["ZERO_DRUG"]["V_WIS"]
        V_WIS_random = result["RANDOM"]["V_WIS"]

        V_WIS_lst.append(V_WIS)
        V_clinician_lst.append(V_clinician)
        V_WIS_zero_drug_lst.append(V_WIS_zero_drug)
        V_WIS_random_lst.append(V_WIS_random)

        if V_WIS > best_policy["V_WIS"]:
            best_policy["value_func"] = value_func
            best_policy["train_df"] = train_df
            best_policy["T"] = T
            best_policy["V_WIS"] = V_WIS
            best_policy["test_indices"] = test_indices
            best_policy["test_states"] = test_states
            best_policy["clinican_action_freq"] = clinican_action_freq
            best_policy["ai_action_freq"] = ai_action_freq
            best_policy["vaso_quantiles"] = vaso_quantiles
            best_policy["iv_quantiles"] = iv_quantiles

    plot_estimated_policy_value(V_clinician_lst, V_WIS_lst, V_WIS_zero_drug_lst, V_WIS_random_lst)


if __name__ == '__main__':

    data_path = sys.argv[1]
    data = pd.read_csv(data_path)

    main(data)


