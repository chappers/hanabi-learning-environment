"""
This is a multi-agent extension so that we can run the
default rllib algorithms over it...

Use this environment as inspiration
https://github.com/ray-project/ray/blob/master/rllib/examples/twostep_game.py
"""

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.qmix.qmix_policy import ENV_STATE
from gym.spaces import Dict, Discrete, Box, Tuple
from hanabi_learning_environment.rl_env import HanabiEnv
from hanabi_learning_environment import pyhanabi


class MultiHanabiEnv(HanabiEnv, MultiAgentEnv):
    def __init__(
        self,
        hanabi_config={
            "colors": 5,
            "ranks": 5,
            "players": 5,
            "max_information_tokens": 8,
            "max_life_tokens": 3,
            "observation_type": pyhanabi.AgentObservationType.MINIMAL.value,
        },
        game_config={},
    ):
        # see the other env
        super(MultiHanabiEnv, self).__init__(config)

        self.config = config
        self.max_actions = self.num_moves() + 1
        self.num_players = config["players"]
        self.global_state_size = self.vectorized_global_shape()[0]  # to do...
        self.state_size = self.vectorized_observation_shape()[0]
        self.global_state_size = self.vectorized_observation_shape()[0] * 4  # naive way

        self.with_state = self.game_config.get(
            "with_state", False
        )  # true if we use QMIX, if vectorised env it is True
        self.centralised_critic = self.game_config.get("centralised_critic", False)
        self.sparse_reward = game_config.get("sparse_reward", True)

        if not self.with_state:
            self.observation_space = Dict(
                {
                    "action_mask": Box(-10, 10, shape=(self.max_actions,)),
                    "obs": Box(-3e38, 3e38, shape=(state_size,)),
                }
            )
        else:
            self.observation_space = Dict(
                {
                    "action_mask": Box(
                        -10, 10, shape=(self.max_actions,)
                    ),  # this is based on legal_moves_as_int
                    "obs": Box(
                        -3e38, 3e38, shape=(state_size,)
                    ),  # we need to encode this ourselves
                    ENV_STATE: Box(
                        -3e38, 3e38, shape=(self.global_state_size,)
                    ),  # this is output of obs_dict['vectorised']
                }
            )

        if self.centralised_critic:
            raise NotImplemented

    def reset(self):
        obs = super().reset()
        return self._obs()

    def step(self, action):
        """MARL environments expect action from all players"""
        obs, rwd, done, info = super().step(action[self.state.cur_player()])
        obs_all = self._obs()
        dones = {"__all__": done}
        rewards = {idx: rwd for idx in range(self.num_players)}
        return obs_all, rewards, dones, {}

    def _obs(self):
        obs = self._make_observation_all_players()
        obs_state = [x["vectorized"] for x in obs["player_observation"]]
        action_mask_int = [x["legal_moves_as_int"] for x in obs["player_observation"]]
        action_mask = []
        for legal_moves in action_mask_int:
            mv = np.zeros(self.max_actions)
            if len(legal_moves) > 0:
                for lm in legal_moves:
                    mv[lm + 1] = 1
            else:
                mv[0] = 1
            action_mask.append(mv.copy())

        global_state = np.concatenate(obs_state)
        # buid up the obs...
        obs_unit = {}
        for idx in range(self.num_players):
            obs_unit[idx] = {
                "obs": obs_state[idx],
                "action_mask": action_mask[idx],
                ENV_STATE: global_state,
            }
        return obs_unit

        if with_state:

            # unit_loc = self.get_unit_locations() # we skip this as its not used right now
            # it could be embedded as part of obs - as down in SMAC environment.
            return {"obs": obs, "action_mask": action_mask, ENV_STATE: global_state}

    def vectorized_global_shape(self):
        # just one-hot encode everything...
        # we need to be careful with that state should only
        # be included based on clues only - not revealing extra info
        life_tokens = self.config["max_life_tokens"]
        information_tokens = self.config["max_information_tokens"]
        total_cards = self.config["ranks"] * self["colors"] * 4

        # we can duplicate this - or perform a count
        # neural net shouldn't care about it...
        # card order in tensor

        # - not seen: inferred if missing...
        # - player holds
        # - on board
        # - discarded
        return life_tokens + information_tokens + total_cards


config = {
    "colors": 5,
    "ranks": 5,
    "players": 2,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
    "observation_type": pyhanabi.AgentObservationType.MINIMAL.value,
}
game = pyhanabi.HanabiGame(config)
hb = MultiHanabiEnv(config)
obs = hb.reset()
obs["player_observations"][0].keys()
obs["player_observations"][0]["pyhanabi"]
