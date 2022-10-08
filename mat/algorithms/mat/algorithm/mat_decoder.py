import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical, Normal
from mat.algorithms.utils.util import check, init


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        logit, v_loc = decoder(shifted_action, obs_rep, obs)
        logit = logit[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log, v_loc


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit, v_loc = decoder(shifted_action, obs_rep, obs)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy, v_loc


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean, v_loc = decoder(shifted_action, obs_rep, obs)
        act_mean = act_mean[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log, v_loc


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean, v_loc = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy, v_loc



class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class DecodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type == 'Discrete':
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                nn.GELU())
        else:
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))
        self.val_head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, 1)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        obs_embeddings = self.obs_encoder(obs)
        action_embeddings = self.action_encoder(action)
        x = action_embeddings
        x = self.ln(x)
        for block in self.blocks:
            x = block(x, obs_embeddings)
        logit = self.head(x)
        val = self.val_head(x)

        return logit, val


class MultiAgentDecoder(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentDecoder, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy, v_loc = discrete_parallel_act(self.decoder, None, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy, v_loc = continuous_parallel_act(self.decoder, None, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)
        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        if self.action_type == "Discrete":
            output_action, output_action_log, v_loc = discrete_autoregreesive_act(self.decoder, None, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log, v_loc = continuous_autoregreesive_act(self.decoder, None, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        _, __, v_loc = self.get_actions(state, obs, available_actions)

        return v_loc



