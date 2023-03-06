import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.actor import DeterministicPolicy, TanhGaussianPolicy, CategoricalPolicy
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp


class Actor_RNN(nn.Module):
    TD3_name = "td3"
    BCD_name = "bcd"  # Behavior cloning for discrete actions
    BCD_SACD_name = "bcd-sacd"  # Behavior cloning for discrete actions + SACD
    BCD2SACD_name = "bcd2sacd"  # Behavior cloning then SACD
    BC2SAC_name = "bc2sac"  # Behavior cloning then SAC
    SAC_name = "sac"
    BC_name = "bc"  # Behavior cloning for continuous actions
    BC_SAC_name = "bc-sac"  # Behvior cloning for continuous actions + SAC
    SACE_name = "sace"  # Current method for continuous actions
    SACA_name = "saca"  # Advisor off-policy for continuous actions
    SACD_name = "sacd"
    SACDA_name = "sacda"  # Advisor off-policy for discrete actions
    SACDE_name = "sacde"  # Current method for discrete actions
    SACDF_name = "sacdf"  # Current method for discrete actions but reducing alpha
    LSTM_name = "lstm"
    GRU_name = "gru"
    RNNs = {
        LSTM_name: nn.LSTM,
        GRU_name: nn.GRU,
    }

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo,
        action_embedding_size,
        state_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        assert algo in [self.TD3_name, self.SAC_name, self.SACE_name,
                        self.SACA_name, self.SACD_name, self.SACDE_name,
                        self.SACDA_name, self.BC_SAC_name, self.BCD2SACD_name,
                        self.BC2SAC_name, self.BCD_name, self.BCD_SACD_name,
                        self.BC_name, self.SACDF_name]
        self.algo = algo

        ### Build Model
        ## 1. embed action, state (Feed-forward layers first)
        if isinstance(obs_dim, tuple):
            if obs_dim == (84, 84, 3):
                # Block picking observation
                self.state_encoder = utl.BlockStackingObsFeatureExtractor(obs_dim)
            else:
                raise NotImplementedError
            state_embedding_size = self.state_encoder.get_output_size()
        else:
            self.state_encoder = utl.FeatureExtractor(obs_dim, state_embedding_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(
            action_dim, action_embedding_size, F.relu
        )

        ## 2. build RNN model
        rnn_input_size = (
            action_embedding_size + state_embedding_size
        )
        self.rnn_hidden_size = rnn_hidden_size

        assert encoder in self.RNNs
        self.encoder = encoder
        self.num_layers = rnn_num_layers

        self.rnn = self.RNNs[encoder](
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
            bias=True,
        )
        # never add activation after GRU cell, cuz the last operation of GRU is tanh

        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        ## 3. build another obs branch
        if isinstance(obs_dim, tuple):
            if obs_dim == (84, 84, 3):
                # Block Picking observation
                self.current_state_encoder = utl.BlockStackingObsFeatureExtractor(obs_dim)
            else:
                raise NotImplementedError
        else:
            self.current_state_encoder = utl.FeatureExtractor(
                obs_dim, state_embedding_size, F.relu
            )

        ## 4. build policy
        if self.algo == self.TD3_name:
            self.policy = DeterministicPolicy(
                obs_dim=self.rnn_hidden_size + state_embedding_size,
                action_dim=self.action_dim,
                hidden_sizes=policy_layers,
            )

        elif self.algo in [self.SAC_name, self.SACE_name,
                           self.SACA_name, self.BC_name,
                           self.BC_SAC_name, self.BC2SAC_name]:
            self.policy = TanhGaussianPolicy(
                obs_dim=self.rnn_hidden_size + state_embedding_size,
                action_dim=self.action_dim,
                hidden_sizes=policy_layers,
            )

            if self.algo in self.SACA_name:
                self.aux_policy = TanhGaussianPolicy(
                    obs_dim=self.rnn_hidden_size + state_embedding_size,
                    action_dim=self.action_dim,
                    hidden_sizes=policy_layers,
                )

        else:  # SACD, BCD, SACD-A, BCD-SACD
            self.policy = CategoricalPolicy(
                obs_dim=self.rnn_hidden_size + state_embedding_size,
                action_dim=self.action_dim,
                hidden_sizes=policy_layers,
            )

            if self.algo == self.SACDA_name:
                self.aux_policy = CategoricalPolicy(
                    obs_dim=self.rnn_hidden_size + state_embedding_size,
                    action_dim=self.action_dim,
                    hidden_sizes=policy_layers,
                )

    def get_hidden_states(
        self, prev_actions, observs, initial_internal_state=None
    ):
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_encoder(prev_actions)
        input_s = self.state_encoder(observs)
        inputs = torch.cat((input_a, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # initial_internal_state is zeros
            output, _ = self.rnn(inputs)
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.rnn(inputs, initial_internal_state)
            return output, current_internal_state

    def forward(self, prev_actions, observs):
        """
        For prev_actions a, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        assert prev_actions.dim() == observs.dim() == 3
        assert prev_actions.shape[0] == observs.shape[0]

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with states
        # return the hidden states (T+1, B, dim)
        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, observs=observs
        )

        # 2. another branch for current obs
        curr_embed = self.current_state_encoder(observs)  # (T+1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_states, curr_embed), dim=-1)  # (T+1, B, dim)

        # 4. Actor
        if self.algo == self.TD3_name:
            new_actions = self.policy(joint_embeds)
            return new_actions, None  # (T+1, B, dim), None
        elif self.algo == self.SAC_name:
            new_actions, _, _, log_probs = self.policy(
                joint_embeds, return_log_prob=True
            )
            return new_actions, log_probs  # (T+1, B, dim), (T+1, B, 1)
        elif self.algo in [self.BC_SAC_name, self.BC2SAC_name]:
            new_actions, means, _, log_probs = self.policy(
                joint_embeds, return_log_prob=True
            )
            return new_actions, means, log_probs
        elif self.algo == self.SACE_name:
            new_actions, means, log_stds, _ = self.policy(
                joint_embeds, return_log_prob=False
            )
            return new_actions, means, log_stds  # (T+1, B, dim), (T+1, B, 1)

        elif self.algo == self.BC_name:
            new_actions, means, log_stds, _ = self.policy(
                joint_embeds, return_log_prob=False
            )
            return new_actions, means, log_stds

        elif self.algo == self.SACA_name:
            new_actions, means, _, log_probs = self.policy(
                joint_embeds, return_log_prob=True
            )

            _, aux_means, _, _ = self.aux_policy(
                joint_embeds, return_log_prob=False
            )

            return new_actions, log_probs, means, aux_means  # (T+1, B, dim), (T+1, B, 1)
        else:  # SAC-D, BCD, SACD-A, BCD-SACD
            _, probs, log_probs = self.policy(joint_embeds, return_log_prob=True)

            if self.algo == self.SACDA_name:
                _, aux_probs, _ = self.aux_policy(joint_embeds)
            else:
                aux_probs = None

            return probs, log_probs, aux_probs  # (T+1, B, dim), (T+1, B, dim)

    @torch.no_grad()
    def get_initial_info(self):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()

        hidden_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
        if self.encoder == self.GRU_name:
            internal_state = hidden_state
        else:
            cell_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
            internal_state = (hidden_state, cell_state)

        return prev_action, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        obs,
        deterministic=False,
        return_log_prob=False,
        exploration_noise=0.0,
    ):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module

        # 1. get hidden state and current internal state
        ## NOTE: in T=1 step rollout (and RNN layers = 1), for GRU they are the same,
        # for LSTM, current_internal_state also includes cell state, i.e.
        # hidden state: (1, B, dim)
        # current_internal_state: (layers, B, dim) or ((layers, B, dim), (layers, B, dim))
        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            observs=obs,
            initial_internal_state=prev_internal_state,
        )
        # 2. another branch for current obs
        curr_embed = self.current_state_encoder(obs)  # (1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_state, curr_embed), dim=-1)  # (1, B, dim)
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        if self.algo == self.TD3_name:
            mean = self.policy(joint_embeds)
            if deterministic:
                action_tuple = (mean, mean, None, None)
            else:
                action = (mean + torch.randn_like(mean) * exploration_noise).clamp(
                    -1, 1
                )  # NOTE
                action_tuple = (action, mean, None, None)
        elif self.algo in [self.SAC_name, self.SACE_name,
                           self.SACA_name, self.BC_name,
                           self.BC_SAC_name, self.BC2SAC_name]:
            action_tuple = self.policy(
                joint_embeds, False, deterministic, return_log_prob
            )
        else:
            # sac-discrete
            action, prob, log_prob = self.policy(
                joint_embeds, deterministic, return_log_prob
            )
            action_tuple = (action, prob, log_prob, None)
        return action_tuple, current_internal_state
