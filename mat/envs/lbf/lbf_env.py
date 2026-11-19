import gymnasium
import lbforaging  # noqa: F401, registers the Foraging-* environments with gymnasium
import numpy as np
from gym.spaces import Box, Discrete


class LBFEnv:
    """Level-Based Foraging (https://github.com/semitable/lb-foraging).

    `scenario` is the id of a registered LBF environment, e.g. "Foraging-8x8-2p-2f-coop-v3".
    All agents receive the sum of the agents' rewards, so the episode return is the common
    return reported in LBF benchmarks. The shared observation is the concatenation of all
    agents' observations. All actions are available, LBF replaces invalid actions with NONE.
    """

    def __init__(self, scenario):
        self.env = gymnasium.make(scenario)
        self.n_agents = len(self.env.action_space)
        self._seed = None

        obs_dim = self.env.observation_space[0].shape[0]
        self.n_actions = self.env.action_space[0].n
        self.observation_space = [Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                                  for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-np.inf, high=np.inf, shape=(self.n_agents * obs_dim,),
                                            dtype=np.float32) for _ in range(self.n_agents)]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]

    def seed(self, seed):
        # applied on the next reset
        self._seed = seed

    def _process_obs(self, obs):
        obs = np.stack(obs).astype(np.float32)
        share_obs = np.tile(obs.reshape(1, -1), (self.n_agents, 1))
        available_actions = np.ones((self.n_agents, self.n_actions), dtype=np.float32)
        return obs, share_obs, available_actions

    def reset(self):
        obs, _ = self.env.reset(seed=self._seed)
        self._seed = None
        return self._process_obs(obs)

    def step(self, actions):
        actions = [int(a) for a in np.asarray(actions).reshape(-1)]
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        obs, share_obs, available_actions = self._process_obs(obs)
        team_reward = float(np.sum(rewards))
        rewards = [[team_reward]] * self.n_agents
        dones = np.array([terminated or truncated] * self.n_agents, dtype=bool)
        infos = [info] * self.n_agents
        return obs, share_obs, rewards, dones, infos, available_actions

    def render(self, mode="human"):
        return self.env.unwrapped.render(mode=mode)

    def close(self):
        self.env.close()
