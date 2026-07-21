#!/usr/bin/env python

import argparse
import RNA
from typing import cast, List, Tuple
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

from gfn.containers import Trajectories
from gfn.gym import RNAPairingEnv
from gfn.gym import RNAStackingEnv, RNAPseudoknotStackingEnv, RNANonPseudoknotStackingEnv
from gfn.estimators import DiscretePolicyEstimator, Estimator, ScalarEstimator
from gfn.gflownet import TBGFlowNet
from gfn.preprocessors import KHotPreprocessor, Preprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.utils.training import validate
from collections import Counter
import faulthandler
faulthandler.enable()


class RND(nn.Module):
    """
    Random Network Distillation (RND) module.
    Predicts the random target net from the state.
    """

    def __init__(
        self,
        state_dim: int,
        preprocessor: Preprocessor,
        reward_scale: float = 0.1,
        loss_scale: float = 0.1,
        hidden_dim: int = 256,
        s_latent_dim: int = 128,
    ) -> None:
        """
        Args:
            state_dim: The dimension of the state space.
            preprocessor: The preprocessor for the state space.
            reward_scale: The scale of the reward.
            loss_scale: The scale of the loss.
            hidden_dim: The dimension of the hidden layer.
            s_latent_dim: The dimension of the latent state.
        """
        super().__init__()
        self.preprocessor = preprocessor
        self.reward_scale = reward_scale
        self.loss_scale = loss_scale

        self.random_target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )

        self.predictor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )

    def forward(self, states: States) -> torch.Tensor:
        l2_error = torch.zeros(states.batch_shape, device=states.device)
        valid_states = states[~states.is_sink_state]

        states_tensor = self.preprocessor(valid_states).float()
        random_target_feature = self.random_target_net(states_tensor).detach()
        predictor_feature = self.predictor_net(states_tensor)

        l2_error[~states.is_sink_state] = torch.norm(
            random_target_feature - predictor_feature, dim=-1, p=2
        )
        return l2_error

    def compute_intrinsic_reward(self, states: States) -> torch.Tensor:
        l2_error = self(states)
        return l2_error.detach() * self.reward_scale

    def compute_rnd_loss(self, states: States) -> torch.Tensor:
        l2_error = self(states)
        l2_error = l2_error.sum(dim=0) / (~states.is_sink_state).sum(dim=0)
        return l2_error.mean() * self.loss_scale


class TBGAFN(TBGFlowNet):
    """
    Generative Augmented Flow Networks based on the Trajectory Balance loss.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator,
        rnd: RND,
        logZ: nn.Parameter | ScalarEstimator | None = None,
        init_logZ: float = 0.0,
        use_edge_ri: bool = False,
        flow_estimator: ScalarEstimator | None = None,
        log_reward_clip_min: float = -float("inf"),
    ):
        super().__init__(pf, pb, logZ, init_logZ, log_reward_clip_min)
        self.rnd = rnd
        self.use_edge_ri = use_edge_ri
        if use_edge_ri and flow_estimator is None:
            raise ValueError("flow_estimator is required if use_edge_ri is True")
        self.flow_estimator = flow_estimator

    def rnd_parameters(self) -> list[torch.Tensor]:
        return list(self.rnd.parameters())

    def flow_parameters(self) -> list[torch.Tensor]:
        if self.flow_estimator is None:
            return []
        return list(
            [v for k, v in self.flow_estimator.named_parameters() if "trunk" not in k]
        )

    def get_scores(
        self, trajectories: Trajectories, recalculate_all_logprobs: bool = True
    ) -> torch.Tensor:
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        log_rewards = trajectories.log_rewards
        assert log_rewards is not None

        if self.use_edge_ri:
            assert self.flow_estimator is not None

            log_state_flows = torch.zeros(
                trajectories.states.batch_shape, device=trajectories.states.device
            )
            log_state_flows[~trajectories.states.is_sink_state] = self.flow_estimator(
                trajectories.states[~trajectories.states.is_sink_state]
            ).squeeze(-1)

            edge_ri = torch.zeros(
                trajectories.states.batch_shape, device=trajectories.states.device
            )
            edge_ri[~trajectories.states.is_sink_state] = self.rnd.compute_intrinsic_reward(
                trajectories.states[~trajectories.states.is_sink_state]
            )

            terminal_ri = edge_ri[
                trajectories.terminating_idx - 1,
                torch.arange(trajectories.n_trajectories, device=edge_ri.device),
            ]
            _terminal_part = torch.stack([log_rewards, terminal_ri.log()], dim=0).logsumexp(dim=0)
            _interm_part = torch.stack(
                [log_pb_trajectories, edge_ri[1:].log() - log_state_flows[1:]], dim=0
            ).logsumexp(dim=0)
            log_target = _terminal_part + _interm_part.sum(dim=0)
        else:
            terminal_ri = self.rnd.compute_intrinsic_reward(
                trajectories.terminating_states
            )
            _terminal_part = torch.stack([log_rewards, terminal_ri.log()], dim=0).logsumexp(dim=0)
            log_target = _terminal_part + log_pb_trajectories.sum(dim=0)

        scores = log_pf_trajectories.sum(dim=0) - log_target
        return scores

    def loss(
        self,
        env: RNAPseudoknotStackingEnv,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        loss = super().loss(env, trajectories, recalculate_all_logprobs, reduction)
        rnd_loss = self.rnd.compute_rnd_loss(trajectories.states)
        return loss + rnd_loss


def is_valid_pair(a, b):
    return (a == 'A' and b == 'U') or (a == 'U' and b == 'A') or \
           (a == 'G' and b == 'C') or (a == 'C' and b == 'G') or \
           (a == 'G' and b == 'U') or (a == 'U' and b == 'G')


def enumerate_base_pairs(seq):
    pairs = []
    for i in range(len(seq)):
        for j in range(i + 4, len(seq)):
            if is_valid_pair(seq[i], seq[j]):
                pairs.append((i, j))
    return pairs


def filter_unstackable_pairs(all_possible_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    pairs_set = set(all_possible_pairs)
    stackable_pairs_list = []
    for i, j in all_possible_pairs:
        can_be_stacked = (i - 1, j + 1) in pairs_set or (i + 1, j - 1) in pairs_set
        if can_be_stacked:
            stackable_pairs_list.append((i, j))
    return stackable_pairs_list

def enumerate_stacks(pairs):
    stacks = []
    for outer1, outer2 in pairs:
        for inner1, inner2 in pairs:
            if outer1 + 1 == inner1 and inner2 + 1 == outer2:
                stacks.append([(outer1, outer2), (inner1, inner2)])
    return stacks


class LogZModule(nn.Module):
    def __init__(self, logZ_param=None):
        super().__init__()
        if logZ_param is not None:
            self.logZ_param = logZ_param
        else:
            self.logZ_param = nn.Parameter(torch.tensor(0.0))
        self.input_dim = 1

    def forward(self, x):
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
        return self.logZ_param.expand(batch_size, 1)

class EpsilonExponentialScheduler:
    def __init__(self, initial_epsilon, min_epsilon, decay_factor):
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_factor = decay_factor

    def get_epsilon(self):
        # 指数関数的にepsilonを減少させる
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)
        return self.epsilon

def print_structure_frequencies(
    env,
    sampler,
    fc,
    RT,
    batch_size,
    n_samples=1000
):
    structures = []
    energies = []

    num_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(num_batches):
        sampled_trajectories = sampler.sample_trajectories(
            env,
            n=batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=0.0,
        )

        for state in sampled_trajectories.terminating_states.tensor:
            structure = env.state_tensor_to_dotbracket(state)
            structures.append(structure)

            energy = fc.eval_structure(structure)
            energies.append(np.exp(-energy / RT))

    structure_counts = Counter(structures)

    structure_score_dict = {}
    for structure, score in zip(structures, energies):
        if structure not in structure_score_dict:
            structure_score_dict[structure] = []
        structure_score_dict[structure].append(score)

    print("\n=== Sampled Structure Frequencies ===")
    for structure, count in structure_counts.most_common():
        avg_score = np.mean(structure_score_dict[structure])
        print(
            f"{structure}: "
            f"{count} samples, "
            f"score={avg_score:.2f}"
        )

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # seq = "UGGGAAAACCC"
    # seq = "GGAAGGAGGAACCUCCUCC"
    # seq = "GGUCGGAAAUUCCCGGAUUUGGAACCUCGG"
    # seq = "GGGAAAUCCCGGAUUUCCCGGAUUAACCGG"
    seq = "AACUAAAACAAUUUUUGAAGAACAGUUUCUGUACUUCAUUGGUAUGUAGAGACUUC"
    valid_pairs = enumerate_base_pairs(seq)
    # valid_pairs = filter_unstackable_pairs(valid_pairs)   #スタックを形成しうるペアに限定
    stacks = enumerate_stacks(valid_pairs)

    print("valid pairs: ", valid_pairs)
    print("stack:", stacks)

    fc = RNA.fold_compound(seq)
    G = fc.pf()
    k = 1.9872036e-3
    T = args.T_min
    RT = k * T
    if isinstance(G, list):
        G = G[1]

    logZ = -G / RT
    Z = np.exp(logZ)
    logZ = np.log(Z)
    print(f"Partition function Z: {Z:.2e}, logZ: {logZ:.2f}")

    logZ_module = LogZModule(logZ_param=logZ)
    #RTを今の温度の値に設定
    RT = args.T0   

    min_epsilon = 0.1
    decay_factor = 0.999

    scheduler = EpsilonExponentialScheduler(args.epsilon, min_epsilon, decay_factor)

    # env = RNAPairingEnv(seq, valid_pairs, device=device)   #スタックを形成しうるペアに限定
    env = RNAStackingEnv(seq, stacks, device=device)
    preprocessor = KHotPreprocessor(height=len(seq)+1, ndim=len(seq))

    module_PF = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions, activation_fn="leaky_relu")
    if not args.uniform_pb:
        module_PB = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1, trunk=module_PF.trunk)
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True)

    rnd = RND(state_dim=preprocessor.output_dim, preprocessor=preprocessor,
              reward_scale=args.rnd_reward_scale, loss_scale=args.rnd_loss_scale,
              hidden_dim=args.rnd_hidden_dim, s_latent_dim=args.rnd_s_latent_dim)

    flow_estimator = None
    if args.use_edge_ri:
        flow_estimator = ScalarEstimator(
            module=MLP(input_dim=preprocessor.output_dim, output_dim=1, trunk=module_PF.trunk),
            preprocessor=preprocessor
        )

    gflownet = TBGAFN(
        pf=pf_estimator, pb=pb_estimator, init_logZ=logZ, rnd=rnd,
        use_edge_ri=args.use_edge_ri, flow_estimator=flow_estimator
    )
    ref_logZ_value = float(logZ)
    gflownet.logZ.data.fill_(ref_logZ_value)
    gflownet.logZ.requires_grad = True

    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(device)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.rnd_parameters(), "lr": args.lr_rnd})
    if args.use_edge_ri:
        optimizer.add_param_group({"params": gflownet.flow_parameters(), "lr": args.lr})
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})
    # optimizer.add_param_group({
    # "params": gflownet.logz_parameters(),
    # "lr": args.lr_logz
    # })
    visited_terminating_states = env.states_from_batch_shape((0,))
    T_min = args.T_min
    T_decay = args.T_decay

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        if(it > 15000):
            env.temperature = max(
                T_min,
                T * (T_decay ** (it - 10000))
            )
            RT = k * env.temperature
        epsilon = scheduler.get_epsilon()
        if it % 50 ==0:
            print(f"Step {it}: Epsilon = {epsilon:4f}")
        trajectories = sampler.sample_trajectories(env, n=args.batch_size, save_logprobs=True, save_estimator_outputs=False, epsilon=epsilon)
        visited_terminating_states.extend(cast(DiscreteStates, trajectories.terminating_states))

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()
        optimizer.step()

        if (it + 1) % 1000 == 0:
            print(f"\n===== Iteration {it+1} =====")
            print_structure_frequencies(
                env,
                sampler,
                fc,
                RT,
                args.batch_size,
                n_samples=1000
            )

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "T": f"{env.temperature:.2f}"
        })

    # --- Sampling and counting structures ---
    n_samples = 10000
    num_batches = (n_samples + args.batch_size - 1) // args.batch_size
    structures = []
    energies = []

    for it in range(num_batches):
        current_batch_size = min(args.batch_size, n_samples - len(structures))
        if current_batch_size <= 0:
            break
        sampled_trajectories = sampler.sample_trajectories(env, n=args.batch_size, save_logprobs=False, save_estimator_outputs=False, epsilon=0.0)

        states = sampled_trajectories.terminating_states.tensor
        for state in states:
            structure = env.state_tensor_to_dotbracket(state)
            structures.append(structure)
            energy = fc.eval_structure(structure)
            energies.append(np.exp(-energy/RT))
            # energies.append(-energy)

    structure_counts = Counter(structures)
    structure_energy_dict = {}
    for structure, energy in zip(structures, energies):
        if structure not in structure_energy_dict:
            structure_energy_dict[structure] = []
        structure_energy_dict[structure].append(energy)

    print("\n=== Sampled Structure Frequencies ===")
    for structure, count in structure_counts.most_common():
        avg_energy = sum(structure_energy_dict[structure]) / len(structure_energy_dict[structure])
        print(f"{structure}: {count} samples, score: {avg_energy:.2f} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the estimators' modules")
    parser.add_argument("--lr_logz", type=float, default=1e-1, help="Learning rate for the logZ parameter")
    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform backward policy")
    parser.add_argument("--n_iterations", type=int, default=100000, help="Number of iterations")
    parser.add_argument("--validation_interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--validation_samples", type=int, default=100, help="Number of validation samples to evaluate PMF")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_edge_ri", default=True, help="Use edge-based intrinsic rewards")
    parser.add_argument("--lr_rnd", type=float, default=1e-4, help="Learning rate for the RND module")
    parser.add_argument("--rnd_reward_scale", type=float, default=1.0, help="Reward scale for RND")
    parser.add_argument("--rnd_loss_scale", type=float, default=1.0, help="Loss scale for RND")
    parser.add_argument("--rnd_hidden_dim", type=int, default=256, help="Hidden layer dim for RND")
    parser.add_argument("--rnd_s_latent_dim", type=int, default=128, help="Latent state dim for RND")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the sampler")
    #焼なまし法用
    parser.add_argument("--T0", type=float, default=400.0, help="Initial temperature")
    parser.add_argument("--T_min", type=float, default=310.15, help="Minimum temperature")
    parser.add_argument("--T_decay", type=float, default=0.99995, help="decay factor of temperature")
    args = parser.parse_args()

    main(args)
