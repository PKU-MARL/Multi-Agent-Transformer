from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
import gfootball.env as football_env
from .encode.obs_encode import FeatureEncoder
from .encode.rew_encode import Rewarder

from .multiagentenv import MultiAgentEnv


class FootballEnv(MultiAgentEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scenario = kwargs["env_args"]["scenario"]
        self.n_agents = kwargs["env_args"]["n_agent"]
        self.reward_type = kwargs["env_args"]["reward"]
        self.env = football_env.create_environment(env_name=self.scenario,
                                                   number_of_left_players_agent_controls=self.n_agents,
                                                   representation="raw",
                                                   # representation="simple115v2",
                                                   rewards=self.reward_type)
        self.feature_encoder = FeatureEncoder()
        self.reward_encoder = Rewarder()

        self.action_space = [gym.spaces.Discrete(self.env.action_space.nvec[1]) for n in range(self.n_agents)]

        tmp_obs_dicts = self.env.reset()
        tmp_obs = [self._encode_obs(obs_dict)[0] for obs_dict in tmp_obs_dicts]
        self.observation_space = [Box(low=float("-inf"), high=float("inf"), shape=tmp_obs[n].shape, dtype=np.float32)
                                  for n in range(self.n_agents)]
        self.share_observation_space = self.observation_space.copy()

        self.pre_obs = None

    def _encode_obs(self, raw_obs):
        obs = self.feature_encoder.encode(raw_obs.copy())
        obs_cat = np.hstack(
            [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
        )
        ava = obs["avail"]
        return obs_cat, ava

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        obs_dicts = self.env.reset()
        self.pre_obs = obs_dicts
        obs = []
        ava = []
        for obs_dict in obs_dicts:
            obs_i, ava_i = self._encode_obs(obs_dict)
            obs.append(obs_i)
            ava.append(ava_i)
        state = obs.copy()
        return obs, state, ava

    def step(self, actions):
        actions_int = [int(a) for a in actions]
        o, r, d, i = self.env.step(actions_int)
        obs = []
        ava = []
        for obs_dict in o:
            obs_i, ava_i = self._encode_obs(obs_dict)
            obs.append(obs_i)
            ava.append(ava_i)
        state = obs.copy()

        rewards = [[self.reward_encoder.calc_reward(_r, _prev_obs, _obs)]
                   for _r, _prev_obs, _obs in zip(r, self.pre_obs, o)]

        self.pre_obs = o

        dones = np.ones((self.n_agents), dtype=bool) * d
        infos = [i for n in range(self.n_agents)]
        return obs, state, rewards, dones, infos, ava

    def render(self, **kwargs):
        # self.env.render(**kwargs)
        pass

    def close(self):
        pass

    def seed(self, args):
        pass

    def get_env_info(self):

        env_info = {"state_shape": self.observation_space[0].shape,
                    "obs_shape": self.observation_space[0].shape,
                    "n_actions": self.action_space[0].n,
                    "n_agents": self.n_agents,
                    "action_spaces": self.action_space
                    }
        return env_info
