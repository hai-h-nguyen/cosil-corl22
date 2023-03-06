# -*- coding: future_fstrings -*-
import os, sys
import time

import math
import numpy as np
import torch
from torch.nn import functional as F
import random
import gym

from .models.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from .models.policy_rnn_shared import ModelFreeOffPolicy_Shared_RNN as Policy_Shared_RNN
from .models.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer

# RNN policy on vector-based task
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import evaluation as utl_eval
from utils import logger


class Learner:
    """
    Learner class for SAC/TD3 x MLP/RNN
    usage: a wide range of POMDP environments and corresponding tasks
    """

    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed

        self.init_env(**env_args)

        self.init_policy(**policy_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)

    def init_env(
        self,
        env_type,
        env_name,
        max_rollouts_per_task=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        report_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        **kwargs
    ):

        # initialize environment
        assert env_type in ["pomdp"]
        self.env_type = env_type

        if self.env_type == "pomdp":  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp

            self.env_name = env_name

            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        else:
            raise ValueError

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False

        if len(self.train_env.observation_space.shape) == 3:
            self.obs_dim = self.train_env.observation_space.shape
        else:
            self.obs_dim = self.train_env.observation_space.shape[0]  # include 1-dim done

        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_policy(self, arch, separate: bool = True, **kwargs):
        # initialize policy
        if arch == "mlp":
            self.policy_arch = "mlp"
            agent_class = Policy_MLP
        else:
            self.policy_arch = "memory"
            if separate == True:
                agent_class = Policy_RNN
            else:
                agent_class = Policy_Shared_RNN
                logger.log("WARNING: YOU ARE USING SHARED ACTOR-CRITIC ARCH !!!!!!!")
        self.agent = agent_class(
            encoder=arch,  # redundant for Policy_MLP
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            **kwargs,
        ).to(ptu.device)

        logger.log(self.agent)

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        num_rollouts_per_iter,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        buffer_type=None,
        **kwargs
    ):

        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if self.policy_arch == "mlp":
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # rnn
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            if buffer_type is None or buffer_type == SeqReplayBuffer.buffer_type:
                buffer_class = SeqReplayBuffer
            elif buffer_type == RAMEfficient_SeqReplayBuffer.buffer_type:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                observation_type=self.train_env.observation_space.dtype,
            )

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter
        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        eval_stochastic=False,
        num_episodes_per_task=1,
        **kwargs
    ):

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self):
        """
        training loop
        NOTE: the main difference from BORel to varibad is changing the alternation
        of rollout collection and model updates: in varibad, one step collection is
        followed by several model updates; while in borel, we collect several **entire**
        trajectories from random sampled tasks for each model updates.
        """

        self._start_training()

        if self.num_init_rollouts_pool > 0:
            logger.log("Collecting initial pool of data..")
            while (
                self._n_env_steps_total
                < self.num_init_rollouts_pool * self.max_trajectory_len
            ):
                self.collect_rollouts(
                    num_rollouts=1,
                    random_actions=True,
                )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log("env steps", self._n_env_steps_total)

            # some methods will schedule weights based on env steps
            self.update_time()

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                perf = self.log()
                if (
                    self.save_interval > 0
                    and self._n_env_steps_total > 0.05 * self.n_env_steps_total
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf)
        self.save_model(current_num_iters, perf)

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """
        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset())
            obs = obs.reshape(1, *obs.shape)
            done_rollout = False

            # get hidden state at timestep=0, None for mlp
            if self.policy_arch == "memory":
                action, internal_state = self.agent.get_initial_info()
                # temporary storage
                obs_list, act_list, e_act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:  # policy takes hidden state as input for rnn, while takes obs for mlp
                    if self.policy_arch == "mlp":
                        action, _, _, _ = self.agent.act(obs, deterministic=False)
                    else:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            obs=obs,
                            deterministic=False,
                        )

                exp_action = ptu.FloatTensor(utl.env_query_expert(self.train_env))
                if not self.act_continuous:
                    exp_action = F.one_hot(
                        exp_action.long(), num_classes=self.act_dim
                    ).float()  # (1, A)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                # add data to policy buffer - (s+, a, r, s'+, term')
                # term ignore time-out scenarios, but record early stopping
                term = (
                    False
                    if "TimeLimit.truncated" in info
                    or steps >= self.max_trajectory_len
                    else done_rollout
                )

                if self.policy_arch == "mlp":
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    e_act_list.append(exp_action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            if self.policy_arch == "memory":  # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                e_act_buffer = torch.cat(e_act_list, dim=0)  # (L, dim)

                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)
                    e_act_buffer = torch.argmax(
                        e_act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    expert_actions=ptu.get_numpy(e_act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                )
                print(
                    f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.policy_arch == "mlp":
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: # all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update_time(self):
        self.agent.update_time(self._n_env_steps_total, self.n_env_steps_total)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):
        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))

        num_steps_per_episode = self.eval_env._max_episode_steps
        observations = None

        # Additional analysis on action selection for Car-Flag-v1 only
        if self.env_name == 'Car-Flag-v1':
            observations = [0, 0, 0]

        for task_idx, task in enumerate(tasks):
            step = 0
            obs = ptu.from_numpy(self.eval_env.reset())  # reset

            obs = obs.reshape(1, *obs.shape)

            if self.policy_arch == "memory":
                action, internal_state = self.agent.get_initial_info()

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for _ in range(num_steps_per_episode):
                    if self.policy_arch == "mlp":
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=deterministic
                        )
                    else:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            obs=obs,
                            deterministic=deterministic,
                        )

                        # Test expert
                        # action = torch.zeros_like(action)
                        # expert_action = self.eval_env.query_expert()[0]
                        # action[:, expert_action] = 1.0

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0)
                    )
                    if self.env_name == 'Car-Flag-v1':
                        converted_action = torch.argmax(action).item()
                        observations[converted_action] += 1
                    running_reward += reward.item()
                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                    # set: obs <- next_obs
                    obs = next_obs.clone()

                    if done_rollout:
                        if "success" in info.keys():
                            success_rate[task_idx] = info["success"]
                        # for all env types, same
                        break

                returns_per_episode[task_idx, episode_idx] = running_reward
            total_steps[task_idx] = step
        return returns_per_episode, success_rate, observations, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular("rl_loss/" + k, v)
        ## gradient norms
        if self.policy_arch == "memory":
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular("rl_loss/" + k, v)
        logger.dump_tabular()

    def log(self):
        # --- log training  ---
        ## set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)

        # --- evaluation ----
        if self.env_type == "pomdp":
            returns_eval, success_rate_eval, observations_eval, total_steps_eval = self.evaluate(
                self.eval_tasks
            )
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    _,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            if self.env_name == 'Car-Flag-v1':
                logger.record_tabular("carflag-act/L", observations_eval[0])
                logger.record_tabular("carflag-act/R", observations_eval[1])
                logger.record_tabular("carflag-act/INFO", observations_eval[2])

            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            logger.record_tabular(
                "metrics/sr_eval_total", np.mean(success_rate_eval)
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )
                logger.record_tabular(
                    "metrics/sr_eval_total_sto", np.mean(success_rate_eval_sto)
                )
        else:
            raise ValueError

        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular(
            "z/fps",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, iter, perf):
        save_path = os.path.join(
            logger.get_dir(), "save", f"agent_{iter}_perf{perf:.3f}.pt"
        )
        torch.save(self.agent.state_dict(), save_path)