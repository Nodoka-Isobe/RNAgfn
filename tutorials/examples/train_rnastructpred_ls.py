#!/usr/bin/env python

import argparse
import RNA
from typing import cast, List, Tuple
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
torch.set_printoptions(threshold=float('inf'))

from gfn.containers import Trajectories
from gfn.gym import RNAPairingEnv
from gfn.gym import RNAStackingEnv
from gfn.estimators import DiscretePolicyEstimator, Estimator, ScalarEstimator
from gfn.gflownet import TBGFlowNet
from gfn.preprocessors import KHotPreprocessor, Preprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.utils.training import validate
from collections import Counter
from gfn.samplers import LocalSearchSampler, StackLocalSearchSampler
import faulthandler
faulthandler.enable()


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

def indices_to_contact_map(state_indices):
    batch_size, seq_len = state_indices.shape
    device = state_indices.device
    
    # 1. 真っ白な行列(L x L)を作成
    contact_map = torch.zeros((batch_size, seq_len, seq_len), device=device)
    
    # 2. ペアあり(値 > 0)を抽出 
    mask = state_indices > 0
    
    # 3. 相手の 0-based index を取得
    partners = torch.where(mask, state_indices - 1, torch.zeros_like(state_indices))
    
    # 4. 行列の (i, partner_i) にドットを打つ
    rows = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 有効なペアの座標を特定して 1.0 を代入
    b_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)[mask]
    r_idx = rows[mask]
    c_idx = partners[mask]
    
    contact_map[b_idx, r_idx, c_idx] = 1.0
    return contact_map

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # seq = "GGAAGGAGGAACCUCCUCC"
    # seq = "UGGGAAAACCC"
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
    T = 310.15
    RT = k * T
    if isinstance(G, list):
        G = G[1]

    logZ = -G / RT
    Z = np.exp(logZ)
    logZ = np.log(Z)

    logZ_module = LogZModule(logZ_param=logZ)

    min_epsilon = 0.1
    decay_factor = 0.999

    scheduler = EpsilonExponentialScheduler(args.epsilon, min_epsilon, decay_factor)

    # env = RNAPairingEnv(seq, valid_pairs, device=device)   #スタックを形成しうるペアに限定
    env = RNAStackingEnv(seq, stacks, device=device)
    preprocessor = KHotPreprocessor(height=4, ndim=len(seq))

    module_PF = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions, activation_fn="leaky_relu")
    if not args.uniform_pb:
        module_PB = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1, trunk=module_PF.trunk)
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True)

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=logZ)

    sampler = LocalSearchSampler(pf_estimator=pf_estimator, pb_estimator=pb_estimator)
    # sampler = LocalSearchSampler(pf_estimator=pf_estimator, pb_estimator=pb_estimator)
    gflownet = gflownet.to(device)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})
    #optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        epsilon = scheduler.get_epsilon()
        if it % 50 ==0:
            print(f"Step {it}: Epsilon = {epsilon:4f}")
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
            epsilon=epsilon,
            n_local_search_loops=args.n_local_search_loops,
            back_ratio=args.back_ratio,
            use_metropolis_hastings=args.use_metropolis_hastings,
        )
        visited_terminating_states.extend(cast(DiscreteStates, trajectories.terminating_states))

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories, recalculate_all_logprobs=False)
        loss.backward()
        optimizer.step()

        # if (it + 1) % args.validation_interval == 0:
        #     validation_info, _ = validate(env, gflownet, args.validation_samples, visited_terminating_states)
        #     print(f"RND loss: {gflownet.rnd.compute_rnd_loss(trajectories.states).item()}")

        pbar.set_postfix({"loss": loss.item()})

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
        # contact_map = indices_to_contact_map(states)
        # print("contact_map = ", contact_map)
        for state in states:
            structure = env.state_tensor_to_dotbracket(state)
            structures.append(structure)
            energy = fc.eval_structure(structure)
            energies.append(np.exp(-energy / RT))
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
        # print(f"{structure}: {count} samples, score: {avg_energy:.2f} ({avg_energy+20:.2f})")
        print(f"{structure}: {count} samples, score: {avg_energy:.2f} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the estimators' modules")
    parser.add_argument("--lr_logz", type=float, default=1e-1, help="Learning rate for the logZ parameter")
    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform backward policy")
    parser.add_argument("--n_iterations", type=int, default=19000, help="Number of iterations")
    parser.add_argument("--validation_interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--validation_samples", type=int, default=1000, help="Number of validation samples to evaluate PMF")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the sampler")

    # Local search parameters.
    parser.add_argument(
        "--n_local_search_loops",
        type=int,
        default=2,
        help="Number of local search loops",
    )
    parser.add_argument(
        "--back_ratio",
        type=float,
        default=0.5,
        help="The ratio of the number of backward steps to the length of the trajectory",
    )
    parser.add_argument(
        "--use_metropolis_hastings",
        action="store_true",
        help="Use Metropolis-Hastings acceptance criterion",
    )
    args = parser.parse_args()

    main(args)

