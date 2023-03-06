""" Recommended Arch
Separate RNN arch is inspired by a popular RL repo
https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/POMDP/common/value_networks.py#L110
which has another branch to encode current state (and action)

Hidden state update functions get_hidden_state() is inspired by varibad encoder 
https://github.com/lmzintgraf/varibad/blob/master/models/encoder.py
"""

import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from utils import helpers as utl
import torchkit.pytorch_utils as ptu
from torchkit.recurrent_critic import Critic_RNN
from torchkit.recurrent_actor import Actor_RNN
from utils import logger


class ModelFreeOffPolicy_Separate_RNN(nn.Module):
    """Recommended Arch
    RNN TD3/SAC (Recurrent Policy) with separate RNNs
            it may have advantages over shared RNN arch
            by avoiding rnn gradient explosion
            and q loss explosion
    the input trajectory include obs,
            and/or action (action_embedding_size != 0).
    depends on the task where partially observation is
    """

    TD3_name = Actor_RNN.TD3_name
    SAC_name = Actor_RNN.SAC_name
    BCD_name = Actor_RNN.BCD_name
    BCD_SACD_name = Actor_RNN.BCD_SACD_name
    BCD2SACD_name = Actor_RNN.BCD2SACD_name
    BC2SAC_name = Actor_RNN.BC2SAC_name
    BC_SAC_name = Actor_RNN.BC_SAC_name
    BC_name = Actor_RNN.BC_name
    SACE_name = Actor_RNN.SACE_name
    SACA_name = Actor_RNN.SACA_name
    SACD_name = Actor_RNN.SACD_name
    SACDE_name = Actor_RNN.SACDE_name
    SACDF_name = Actor_RNN.SACDF_name
    SACDA_name = Actor_RNN.SACDA_name

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo,
        action_embedding_size,
        state_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        policy_layers,
        rnn_num_layers=1,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        # td3 params
        exploration_noise=0.1,
        target_noise=0.2,
        target_noise_clip=0.5,
        # sac params
        entropy_alpha=0.2,
        automatic_entropy_tuning=True,
        target_entropy=None,
        alpha_lr=3e-4,
        # advisor params
        alpha=20.0,
        # bc + sac/sacd params
        w_bc=0.5,
        # bc -> sac/sac-d params
        bc_perc=0.1,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        assert algo in [self.TD3_name, self.SAC_name, self.SACE_name,
                        self.SACA_name, self.BCD_name, self.BC_name,
                        self.BCD_SACD_name, self.BC_SAC_name,
                        self.BCD2SACD_name, self.BC2SAC_name, self.SACDF_name,
                        self.SACD_name, self.SACDE_name, self.SACDA_name]
        self.algo = algo

        if algo in [self.SACDA_name, self.SACA_name]:
            self.alpha = alpha

        if algo in [self.BCD_SACD_name, self.BC_SAC_name]:
            self.w_bc = w_bc

        if algo in [self.BCD2SACD_name, self.BC2SAC_name]:
            self.w_bc = 1.0
            # The percentage of the total timesteps trained with BC
            self.bc_perc = bc_perc

        # Critics
        self.critic = Critic_RNN(
            obs_dim,
            action_dim,
            encoder,
            algo,
            action_embedding_size,
            state_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            rnn_num_layers,
        )

        # target networks
        self.critic_target = deepcopy(self.critic)

        # Actor
        self.actor = Actor_RNN(
            obs_dim,
            action_dim,
            encoder,
            algo,
            action_embedding_size,
            state_embedding_size,
            rnn_hidden_size,
            policy_layers,
            rnn_num_layers,
        )

        if self.algo in [self.TD3_name]:
            # NOTE: td3 has a target policy (actor)
            self.actor_target = deepcopy(self.actor)
            self.exploration_noise = exploration_noise
            self.target_noise = target_noise
            self.target_noise_clip = target_noise_clip
        elif self.algo in [self.BCD_name, self.BC_name]:
            pass
        else:
            ## automatic entropy coefficient tuning (recommended)
            self.automatic_entropy_tuning = automatic_entropy_tuning

            if self.algo == self.SACDF_name:
                self.automatic_entropy_tuning = False

            if self.automatic_entropy_tuning:
                if target_entropy is not None:
                    if self.algo in [self.SAC_name, self.SACE_name, self.SACDE_name, self.SACA_name]:
                        self.target_entropy = float(target_entropy)
                    else:  # sac-discrete: beta * log(|A|)
                        self.target_entropy = float(target_entropy) * np.log(action_dim)
                else:
                    assert self.algo in [self.SAC_name, self.SACA_name,
                                         self.BC_SAC_name, self.BC2SAC_name]
                    self.target_entropy = -float(action_dim)
                self.log_alpha_entropy = torch.zeros(1, requires_grad=True, device=ptu.device)
                self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
                self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
            else:
                if self.algo == self.SACDF_name:
                    self.alpha_entropy = 1.0
                    self.alpha_entropy_schedule = None
                else:
                    self.alpha_entropy = entropy_alpha

        # No critic for behavior cloning
        if self.algo not in [self.BCD_name, self.BC_name]:
            # use separate optimizers
            self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)

    @torch.no_grad()
    def get_initial_info(self):
        return self.actor.get_initial_info()

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        obs = obs.unsqueeze(0)  # (1, B, 1)

        current_action_tuple, current_internal_state = self.actor.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            obs=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            exploration_noise=self.exploration_noise
            if self.algo in [self.TD3_name]
            else 0.0,
        )

        return current_action_tuple, current_internal_state

    def gaussian_distance(self, inputs):
        means_2, means_1 = inputs
        return ((means_1 - means_2)**2).mean(dim=-1, keepdims=True)

    def forward(self, actions, exp_actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        ### 1. Critic loss
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim) including reaction to last obs
            if self.algo in [self.TD3_name]:
                new_actions, _ = self.actor_target(
                    prev_actions=actions, observs=observs
                )
                action_noise = (
                    torch.randn_like(new_actions) * self.target_noise
                ).clamp(-self.target_noise_clip, self.target_noise_clip)
                new_actions = (new_actions + action_noise).clamp(-1, 1)  # NOTE
            elif self.algo == self.SAC_name:
                new_actions, new_log_probs = self.actor(
                    prev_actions=actions, observs=observs
                )
            elif self.algo == self.SACE_name:
                new_actions, means, _ = self.actor(
                    prev_actions=actions, observs=observs
                )
            elif self.algo in [self.BC2SAC_name, self.BC_SAC_name]:
                new_actions, means, new_log_probs = self.actor(
                    prev_actions=actions, observs=observs
                )
            elif self.algo == self.SACA_name:
                new_actions, new_log_probs, means, aux_means = self.actor(
                    prev_actions=actions, observs=observs
                )
            else:
                new_probs, new_log_probs, _ = self.actor(
                    prev_actions=actions, observs=observs
                )

            next_q1, next_q2 = self.critic_target(
                prev_actions=actions,
                observs=observs,
                current_actions=new_actions
                if self.algo in [self.TD3_name, self.SAC_name,
                                 self.SACE_name, self.SACA_name,
                                 self.BC2SAC_name, self.BC_SAC_name]
                else new_probs,
            )  # (T+1, B, 1 or A)

            min_next_q_target = torch.min(next_q1, next_q2)
            min_next_q_target = min_next_q_target[1:]

            if self.algo not in [self.SACE_name, self.TD3_name]:
                new_log_probs = new_log_probs[1:]

            if self.algo in [self.SAC_name, self.SACA_name]:
                min_next_q_target += self.alpha_entropy * (
                    -new_log_probs
                )  # (T+1, B, 1 or A)

            if self.algo in [self.SACE_name]:
                distance = self.gaussian_distance((means[1:], exp_actions[1:]))
                min_next_q_target += self.alpha_entropy * distance.reshape(min_next_q_target.shape)

            if self.algo in [self.SACD_name, self.BCD_SACD_name,
                             self.BCD2SACD_name, self.SACDE_name,
                             self.SACDA_name, self.SACDF_name]:
                if self.algo in [self.SACDE_name, self.SACDF_name]:
                    min_next_q_target += self.alpha_entropy * (
                        new_log_probs.gather(dim=-1, index=exp_actions[1:].long())
                    )  # (T+1, B, 1 or A)
                else:
                    # for SACD, SACDA, BCD_SACD, BCD2SACD
                    min_next_q_target += self.alpha_entropy * (
                        -new_log_probs
                    )  # (T+1, B, 1 or A)

                min_next_q_target = (new_probs[1:] * min_next_q_target).sum(
                    dim=-1, keepdims=True
                )

            # q_target: (T, B, 1)
            q_target = (
                rewards[1:] + (1.0 - dones[1:]) * self.gamma * min_next_q_target
            )  # next q

        # Q(h(t), a(t)) (T, B, 1)
        q1_pred, q2_pred = self.critic(
            prev_actions=actions,
            observs=observs,
            current_actions=actions[1:],
        )  # (T, B, 1 or A)

        if self.algo in [self.SACD_name, self.BCD_SACD_name,
                         self.SACDF_name,
                         self.SACDE_name, self.SACDA_name]:
            stored_actions = actions[1:]  # (T, B, A)
            stored_actions = torch.argmax(
                stored_actions, dim=-1, keepdims=True
            )  # (T, B, 1)
            q1_pred = q1_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)
            q2_pred = q2_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        # No critic in these algorithms
        if self.algo not in [self.BCD_name, self.BC_name]:
            self.critic_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            self.critic_optimizer.step()

        ### 2. Actor loss
        if self.algo in [self.TD3_name]:
            new_actions, _ = self.actor(
                prev_actions=actions, observs=observs
            )  # (T+1, B, A)
        elif self.algo in [self.SAC_name]:
            new_actions, log_probs = self.actor(
                prev_actions=actions, observs=observs
            )  # (T+1, B, A)
        elif self.algo in [self.BC_SAC_name, self.BC2SAC_name]:
            new_actions, means, log_probs = self.actor(
                prev_actions=actions, observs=observs
            )  # (T+1, B, A)
        elif self.algo == self.SACA_name:
            new_actions, log_probs, means, aux_means = self.actor(
                prev_actions=actions, observs=observs
            )  # (T+1, B, A)
        elif self.algo == self.BC_name:
            _, means, _ = self.actor(
                prev_actions=actions, observs=observs
            )  # (T+1, B, A)
        elif self.algo == self.SACE_name:
            new_actions, means, _ = self.actor(
                prev_actions=actions, observs=observs
            )  # (T+1, B, A)
        else:  # SACD, BCD, SACD-A, BCD-SACD
            new_probs, log_probs, aux_probs = self.actor(
                prev_actions=actions, observs=observs
            )  # (T+1, B, A)

        if self.algo in [self.BCD_name, self.BCD_SACD_name, self.BCD2SACD_name]:
            main_distr = torch.distributions.Categorical(new_probs[:-1])
            main_expert_neg_cross_entropy = main_distr.log_prob(exp_actions[1:].long().squeeze(-1)).unsqueeze(-1)
            bc_loss = -main_expert_neg_cross_entropy.sum() / num_valid

        if self.algo in [self.BC_name, self.BC_SAC_name, self.BC2SAC_name]:
            bc_loss = ((means - exp_actions)**2).sum() / num_valid

        if self.algo == self.SACDA_name:
            main_distr = torch.distributions.Categorical(new_probs[:-1])
            main_expert_neg_cross_entropy = main_distr.log_prob(exp_actions[1:].long().squeeze(-1)).unsqueeze(-1)

            aux_distr = torch.distributions.Categorical(aux_probs[:-1])
            distance = -aux_distr.log_prob(exp_actions[1:].long().squeeze(-1)).unsqueeze(-1)

            use_expert_weights = (
                    torch.exp(-self.alpha * distance)
                    * masks
                    .float()
                ).detach()

            w_main_expert_ce_loss = -(
                use_expert_weights * main_expert_neg_cross_entropy
            ).sum() / num_valid

            aux_expert_ce_loss = (distance * masks).sum() / num_valid

            use_rl_weights = 1 - use_expert_weights

        if self.algo == self.SACA_name:
            main_distance = ((means[:-1] - exp_actions[:-1])**2).mean(dim=-1, keepdims=True)
            aux_distance = ((aux_means[:-1] - exp_actions[:-1])**2).mean(dim=-1, keepdims=True)

            use_expert_weights = (
                    torch.exp(-self.alpha * aux_distance)
                    * masks
                    .float()
                ).detach()

            w_main_expert_ce_loss = (
                use_expert_weights * main_distance
            ).sum() / num_valid

            aux_expert_ce_loss = (aux_distance * masks).sum() / num_valid

            use_rl_weights = 1 - use_expert_weights

        q1, q2 = self.critic(
            prev_actions=actions,
            observs=observs,
            current_actions=new_actions
            if self.algo in [self.TD3_name, self.SAC_name,
                             self.BC_SAC_name, self.BC2SAC_name,
                             self.SACA_name, self.SACE_name]
            else new_probs,
        )  # (T+1, B, 1 or A)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1 or A)

        policy_loss = -min_q_new_actions

        policy_loss = policy_loss[:-1]
        if self.algo in [self.SAC_name, self.SACA_name,
                         self.BCD_SACD_name, self.BC_SAC_name,
                         self.BCD2SACD_name,
                         self.SACD_name, self.SACDA_name]:  # Q(h(t), pi(h(t))) + H[pi(h(t))]
            policy_loss += self.alpha_entropy * log_probs[:-1]

        elif self.algo in [self.SACE_name]:
            distance = self.gaussian_distance((means[:-1], exp_actions[1:]))
            policy_loss += self.alpha_entropy * distance.reshape(policy_loss.shape)

        elif self.algo in [self.SACDE_name, self.SACDF_name]:
            policy_loss += -self.alpha_entropy * log_probs[:-1].gather(dim=-1, index=exp_actions[1:].long())

        if self.algo in [self.SACD_name, self.BCD_SACD_name,
                         self.BCD2SACD_name, self.SACDF_name,
                         self.SACDE_name, self.SACDA_name]:  # E_{a\sim \pi}[Q(h,a)]
            policy_loss = (new_probs[:-1] * policy_loss).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)

        # masked policy_loss
        if self.algo in [self.SACDA_name, self.SACA_name]:
            policy_loss = (policy_loss * use_rl_weights * masks).sum() / num_valid
        else:
            policy_loss = (policy_loss * masks).sum() / num_valid

        self.actor_optimizer.zero_grad()
        if self.algo in [self.SACDA_name, self.SACA_name]:
            (policy_loss + w_main_expert_ce_loss + aux_expert_ce_loss).backward()
        elif self.algo in [self.BCD_name, self.BC_name]:
            bc_loss.backward()
        elif self.algo in [self.BCD_SACD_name, self.BC_SAC_name,
                           self.BCD2SACD_name, self.BC2SAC_name]:
            weighted_loss = self.w_bc*bc_loss + (1-self.w_bc)*policy_loss
            weighted_loss.backward()
        else:
            policy_loss.backward()
        self.actor_optimizer.step()

        ### 3. soft update
        self.soft_target_update()

        ### 4. update alpha
        if self.algo in [self.SAC_name, self.SACA_name, self.SACE_name,
                         self.BCD_SACD_name, self.BC_SAC_name,
                         self.BCD2SACD_name, self.BC2SAC_name,
                         self.SACD_name, self.SACDE_name, self.SACDA_name]:
            # extract valid log_probs
            if self.algo in [self.SACD_name, self.BCD_SACD_name,
                             self.BCD2SACD_name, self.SACDA_name]:  # -> negative entropy (T+1, B, 1)
                log_probs = (new_probs[:-1] * log_probs[:-1]).sum(axis=-1, keepdims=True)

            if self.algo in [self.SACDE_name]:  # -> negative entropy (T+1, B, 1)
                log_probs = (new_probs[:-1] * log_probs[:-1].gather(dim=-1, index=exp_actions[1:].long())).sum(axis=-1, keepdims=True)

            if self.algo in [self.SACE_name]:
                means_distance = self.gaussian_distance((means[:-1], exp_actions[1:]))

            with torch.no_grad():
                if self.algo in [self.SAC_name, self.SACA_name,
                                 self.BC_SAC_name, self.BC2SAC_name]:
                    current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                elif self.algo == self.SACE_name:
                    # reuse the name variable but should be means difference
                    current_log_probs = (means_distance.reshape(masks.shape) * masks).sum() / num_valid
                else:
                    current_log_probs = (log_probs * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            if self.automatic_entropy_tuning:
                if self.algo in [self.SACD_name, self.SAC_name, self.SACA_name,
                                 self.BCD_SACD_name, self.BC_SAC_name,
                                 self.BCD2SACD_name, self.BC2SAC_name,
                                 self.SACE_name, self.SACDA_name]:
                    alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                        current_log_probs + self.target_entropy
                    )

                else:  # SACDE
                    alpha_entropy_loss = self.log_alpha_entropy.exp() * (
                        current_log_probs + self.target_entropy
                    )

                optimize_alpha = True

                # Pure BC finished so start optimizing alpha
                if self.algo in [self.BC2SAC_name, self.BCD2SACD_name]:
                    optimize_alpha = (self.w_bc < 1.0)

                if optimize_alpha:
                    self.alpha_entropy_optim.zero_grad()
                    alpha_entropy_loss.backward()
                    self.alpha_entropy_optim.step()

                self.alpha_entropy = self.log_alpha_entropy.exp().item()

        outputs = {}
        if self.algo not in [self.BCD_name, self.BC_name]:
            outputs.update({"qf1_loss": qf1_loss.item()})
            outputs.update({"qf2_loss": qf2_loss.item()})
            outputs.update({"policy_loss": policy_loss.item()})

        if self.algo not in [self.TD3_name, self.BCD_name, self.BC_name]:
            if self.automatic_entropy_tuning:
                outputs.update({"alpha_entropy_loss": alpha_entropy_loss.item()})
                outputs.update({"log_alpha_entropy": self.log_alpha_entropy.detach().item()})

        if self.algo in [self.SAC_name, self.SACE_name, self.SACE_name,
                         self.SACA_name, self.BCD_SACD_name,
                         self.BCD2SACD_name, self.BC_SAC_name,
                         self.BC2SAC_name, self.SACD_name,
                         self.SACDE_name, self.SACDA_name]:
            outputs.update(
                {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}
            )

        if self.algo == self.SACDF_name:
            outputs.update({"alpha": self.alpha_entropy})

        if self.algo in [self.SACDA_name, self.SACA_name]:
            outputs.update({"aux_expert_ce_loss": aux_expert_ce_loss.item(),
                            "w_main_expert_ce_loss": w_main_expert_ce_loss.item()})

        if self.algo in [self.BCD_SACD_name, self.BC_SAC_name,
                         self.BCD2SACD_name, self.BC2SAC_name]:
            outputs.update({"weighted_loss": weighted_loss.item()})
            outputs.update({"weight_bc": self.w_bc})

            if self.algo in [self.BC2SAC_name, self.BCD2SACD_name]:
                outputs.update({"bc_perc": self.bc_perc})

        if self.algo in [self.BCD_name, self.BC_name]:
            outputs.update({"bc_loss": bc_loss.item()})

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo in [self.TD3_name]:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "q_grad_norm": utl.get_grad_norm(self.critic),
            "q_rnn_grad_norm": utl.get_grad_norm(self.critic.rnn),
            "pi_grad_norm": utl.get_grad_norm(self.actor),
            "pi_rnn_grad_norm": utl.get_grad_norm(self.actor.rnn),
        }

    def update_time(self, current_step, total_step):
        if self.algo in [self.BCD2SACD_name, self.BC2SAC_name]:
            if current_step >= self.bc_perc * total_step:
                self.w_bc = 0.0
            else:
                self.w_bc = 1.0

        if self.algo == self.SACDF_name:
            if self.alpha_entropy_schedule is None:
                # self.alpha_entropy_schedule = utl.LinearSchedule(1.0, 0.0, total_step)
                self.alpha_entropy_schedule = utl.ExponentialSchedule(1.0, 1e-6, total_step)
            else:
                self.alpha_entropy = self.alpha_entropy_schedule.value(current_step)

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones, exp_actions = batch["act"], batch["rew"], batch["term"], batch["exp_act"]

        _, batch_size, _ = actions.shape
        if self.algo in [self.SACD_name, self.SACDE_name,
                         self.SACDF_name,
                         self.BCD_SACD_name, self.BCD2SACD_name,
                         self.SACDA_name, self.BCD_name]:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)

        # Discrete actions with experts
        if self.algo in [self.SACDA_name, self.SACDE_name,
                         self.SACDF_name,
                         self.BCD2SACD_name, self.BCD_SACD_name,
                         self.SACD_name, self.BCD_name]:
            exp_actions = torch.cat(
                (ptu.zeros((1, batch_size, 1)), exp_actions), dim=0
            )  # (T+1, B, dim)
        else:
            exp_actions = torch.cat(
                (ptu.zeros((1, batch_size, self.action_dim)).float(), exp_actions), dim=0
            )  # (T+1, B, dim)

        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)

        return self.forward(actions, exp_actions, rewards, observs, dones, masks)
