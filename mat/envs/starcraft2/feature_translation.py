import numpy as np
import copy
import sys
sys.path.append("../../")
from .smac_maps import get_map_params

enable_task_embedding = True

target_num_agent = 27
target_num_enemy = 32
target_action_dim = 38
local_total_dim = 2001
local_ally_feats_dim = 54
local_enemy_feats_dim = 16
local_move_feats_dim = 4
local_own_feats_dim = 81

task_embedding_dim = 28

find_map = {
    "289": "10m_vs_11m",
    "310": "1c3s5z",
    "1054": "25m",
    "1288": "27m_vs_30m",
    "36": "2m_vs_1z",
    "144": "2s3z",
    "37": "2s_vs_1sc",
    "64": "3m",
    "252": "3s5z",
    "268": "3s5z_vs_3s6z",
    "70": "3s_vs_3z",
    "79": "3s_vs_4z",
    "88": "3s_vs_5z",
    "124": "5m_vs_6m",
    "172": "6h_vs_8z",
    "204": "8m",
    "217": "8m_vs_9m",
    "334": "MMM",
    "370": "MMM2",
    "1084": "bane_vs_bane",
    "346": "corridor",
    "479": "so_many_baneling",
}

race_map = {
    "T": 0,
    "P": 1,
    "Z": 2,
}

unified_unit_type_map = {
    "Marine": 0,
    "Medivac": 1,
    "Marauder": 2,
    "Stalker": 3,
    "Zealot": 4,
    "Colossi": 5,
    "Zergling": 6,
    "Baneling": 7,
    "Hydralisk": 8,
    "Spine_Crawler": 9,
}

map_agent_type_num = {
    "10m_vs_11m": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "1c3s5z": [0, 0, 0, 3, 5, 1, 0, 0, 0, 0],
    "25m": [25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "27m_vs_30m": [27, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "2m_vs_1z": [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "2s3z": [0, 0, 0, 2, 3, 0, 0, 0, 0, 0],
    "2s_vs_1sc": [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    "3m": [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "3s5z": [0, 0, 0, 3, 5, 0, 0, 0, 0, 0],
    "3s5z_vs_3s6z": [0, 0, 0, 3, 5, 0, 0, 0, 0, 0],
    "3s_vs_3z": [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    "3s_vs_4z": [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    "3s_vs_5z": [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    "5m_vs_6m": [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6h_vs_8z": [0, 0, 0, 0, 0, 0, 0, 0, 6, 0],
    "8m": [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "8m_vs_9m": [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "MMM": [7, 1, 2, 0, 0, 0, 0, 0, 0, 0],
    "MMM2": [7, 1, 2, 0, 0, 0, 0, 0, 0, 0],
    "bane_vs_bane": [0, 0, 0, 0, 0, 0, 20, 4, 0, 0],
    "corridor": [0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
    "so_many_baneling": [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
}

map_enemy_type_num = {
    "10m_vs_11m": [11, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "1c3s5z": [0, 0, 0, 3, 5, 1, 0, 0, 0, 0],
    "25m": [25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "27m_vs_30m": [30, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "2m_vs_1z": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "2s3z": [0, 0, 0, 2, 3, 0, 0, 0, 0, 0],
    "2s_vs_1sc": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "3m": [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "3s5z": [0, 0, 0, 3, 5, 0, 0, 0, 0, 0],
    "3s5z_vs_3s6z": [0, 0, 0, 3, 6, 0, 0, 0, 0, 0],
    "3s_vs_3z": [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    "3s_vs_4z": [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    "3s_vs_5z": [0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
    "5m_vs_6m": [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6h_vs_8z": [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    "8m": [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "8m_vs_9m": [9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "MMM": [7, 1, 2, 0, 0, 0, 0, 0, 0, 0],
    "MMM2": [8, 1, 3, 0, 0, 0, 0, 0, 0, 0],
    "bane_vs_bane": [0, 0, 0, 0, 0, 0, 20, 4, 0, 0],
    "corridor": [0, 0, 0, 0, 0, 0, 24, 0, 0, 0],
    "so_many_baneling": [0, 0, 0, 0, 0, 0, 0, 32, 0, 0],
}


def find_map_dim(dim):
    return find_map[str(dim)]


def translate_global_state(state):
    ori_shape = np.shape(state)
    return np.zeros((*ori_shape[:-1], get_state_dim()[0]), dtype=np.float32)


def translate_local_obs(obs):
    len_obs = np.shape(obs)[-1]
    map_name = find_map_dim(len_obs)

    if len_obs > local_total_dim:
        print("tlocal_total_dim (%s) too small, obs dim is %s." % (local_total_dim, len(obs)))
        raise NotImplementedError
    elif len_obs <= local_total_dim:
        if isinstance(obs, list):
            # obs = np.array(copy.deepcopy(obs))
            obs = np.array(obs)
        elif isinstance(obs, np.ndarray):
            # obs = copy.deepcopy(obs)
            obs = obs
        else:
            print("unknwon type %s." % type(obs))
            raise NotImplementedError

    map_info = get_map_params(map_name)
    ori_shape = np.shape(obs)
    obs = obs.reshape((-1, ori_shape[-1]))
    new_obs_n = []
    for i in range(len(obs)):
        new_obs_i = single_translate_local(obs[i], map_info)
        if enable_task_embedding:
            task_embedding = gen_task_embedding(map_name)
            new_obs_i = np.concatenate((new_obs_i, task_embedding))
        new_obs_n.append(new_obs_i)
    new_obs = np.vstack(new_obs_n)
    if enable_task_embedding:
        new_obs = new_obs.reshape((*ori_shape[:-1], local_total_dim + task_embedding_dim))
    else:
        new_obs = new_obs.reshape((*ori_shape[:-1], local_total_dim))

    return new_obs


def single_translate_local(ori_obs, map_info):
    # [310, [8, 24], [9, 9], [1, 4], [1, 33]]
    agent_race = map_info["a_race"]
    enemy_race = map_info["b_race"]
    agent_types = np.array(map_info["agent_unified_type"])
    enemy_types = np.array(map_info["enemy_unified_type"])
    ori_num_agents = map_info["n_agents"]
    ori_num_enemies = map_info["n_enemies"]
    ori_unit_type_bits = map_info["unit_type_bits"]

    # dead agents
    if np.all((ori_obs[:len(ori_obs)-ori_num_agents] == 0.)):
        return np.zeros(local_total_dim, dtype=np.float32)

    ori_action_dim = 6 + ori_num_enemies
    last_decode = 0

    # decode original ally features
    ori_ally_feats_dim = 5 + ori_unit_type_bits + ori_action_dim  # 5 fixed field
    if agent_race == "P":
        ori_ally_feats_dim += 1
    end_decode = last_decode + (ori_num_agents-1) * ori_ally_feats_dim
    ori_ally_feats = ori_obs[last_decode:end_decode].reshape((-1, ori_ally_feats_dim))
    last_decode = end_decode
    # print("ori_ally_feats shape: ", np.shape(ori_ally_feats))

    # decode original enemy features
    ori_enemy_feats_dim = 5 + ori_unit_type_bits
    if enemy_race == "P":
        ori_enemy_feats_dim += 1
    end_decode = last_decode + ori_num_enemies * ori_enemy_feats_dim
    ori_enemy_feats = ori_obs[last_decode:end_decode].reshape((-1, ori_enemy_feats_dim))
    last_decode = end_decode
    # print("ori_enemy_feats shape: ", np.shape(ori_enemy_feats))

    # decode original move features
    ori_move_feats_dim = 4
    end_decode = last_decode + ori_move_feats_dim
    ori_move_feats = ori_obs[last_decode:end_decode]
    last_decode = end_decode
    # print("ori_move_feats shape: ", np.shape(ori_move_feats))

    # decode original own features
    ori_own_feats_dim = 5 + ori_unit_type_bits + ori_action_dim + ori_num_agents
    if agent_race == "P":
        ori_own_feats_dim += 1
    end_decode = last_decode + ori_own_feats_dim
    ori_own_feats = ori_obs[last_decode:end_decode]
    # print("ori_own_feats shape: ", np.shape(ori_own_feats))

    # [2001, [26, 54], [32, 16], [1, 4], [1, 81]]
    new_ally_feats = np.zeros((target_num_agent-1, local_ally_feats_dim), dtype=np.float32)
    new_enemy_feats = np.zeros((target_num_enemy, local_enemy_feats_dim), dtype=np.float32)
    new_move_feats = np.zeros(local_move_feats_dim, dtype=np.float32)
    new_own_feats = np.zeros(local_own_feats_dim, dtype=np.float32)

    # encode ally features
    # only alive and visible allies are valid, otherwise all zeros
    valid_idx = np.argwhere(np.any((ori_ally_feats != 0.), axis=-1) > 0).reshape(-1)
    new_ally_feats[valid_idx, :5] = ori_ally_feats[valid_idx, :5]
    idx = 5
    if agent_race == "P":
        new_ally_feats[valid_idx, 5] = ori_ally_feats[valid_idx, idx]
        idx += 1

    if ori_unit_type_bits > 0:
        ori_type_bits = ori_ally_feats[valid_idx, idx:idx+ori_unit_type_bits]
        ori_types = np.argwhere(ori_type_bits == 1.)[:, -1]
    else:
        ori_types = np.zeros(len(valid_idx), dtype=np.int)
    unified_types = agent_types[ori_types]
    new_ally_feats[valid_idx, unified_types+6] = 1.
    idx += ori_unit_type_bits

    new_ally_feats[valid_idx, 16:16+ori_action_dim] = ori_ally_feats[valid_idx, idx:]

    # encode enemy features
    valid_idx = np.argwhere(np.any((ori_enemy_feats != 0.), axis=-1) > 0).reshape(-1)
    new_enemy_feats[valid_idx, :5] = ori_enemy_feats[valid_idx, :5]
    idx = 5
    if enemy_race == "P":
        new_enemy_feats[valid_idx, 5] = ori_enemy_feats[valid_idx, idx]
        idx += 1

    if ori_unit_type_bits > 0:
        ori_type_bits = ori_enemy_feats[valid_idx, idx:idx+ori_unit_type_bits]
        ori_types = np.argwhere(ori_type_bits == 1.)[:, -1]
    else:
        ori_types = np.zeros(len(valid_idx), dtype=np.int)
    unified_types = enemy_types[ori_types]
    new_enemy_feats[valid_idx, unified_types+6] = 1.

    # encode move features
    new_move_feats = ori_move_feats

    # encode own features
    new_own_feats[:5] = ori_own_feats[:5]
    idx = 5
    if agent_race == "P":
        new_own_feats[5] = ori_own_feats[idx]
        idx += 1

    if ori_unit_type_bits > 0:
        ori_type_bits = ori_own_feats[idx:idx+ori_unit_type_bits]
        ori_types = np.argwhere(ori_type_bits == 1.)[0][0]
    else:
        ori_types = 0
    unified_types = agent_types[ori_types]
    new_own_feats[unified_types+6] = 1.
    idx += ori_unit_type_bits

    new_own_feats[16:16+ori_action_dim] = ori_own_feats[idx:idx+ori_action_dim]
    idx += ori_action_dim
    new_own_feats[54:54+ori_num_agents] = ori_own_feats[idx:idx+ori_num_agents]

    new_obs = np.concatenate((new_ally_feats.flatten(),
                              new_enemy_feats.flatten(),
                              new_move_feats.flatten(),
                              new_own_feats.flatten()))
    return new_obs


def gen_task_embedding(map_name):
    map_info = get_map_params(map_name)
    task_embedding = np.zeros(task_embedding_dim, dtype=np.float32)
    task_embedding[0] = map_info["n_agents"] / target_num_agent
    task_embedding[1] = map_info["n_enemies"] / target_num_enemy
    task_embedding[2+race_map[map_info["a_race"]]] = 1.
    task_embedding[5+race_map[map_info["b_race"]]] = 1.
    task_embedding[8:18] = np.array(map_agent_type_num[map_name]) / target_num_agent
    task_embedding[18:28] = np.array(map_enemy_type_num[map_name]) / target_num_enemy
    return task_embedding


def get_obs_dim():
    if enable_task_embedding:
        return [local_total_dim + task_embedding_dim]
    else:
        return [local_total_dim]


def get_state_dim():
    return get_obs_dim()


def get_act_dim():
    return target_action_dim


def get_num_agents():
    return target_num_agent

