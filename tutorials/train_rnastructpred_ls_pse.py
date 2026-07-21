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
from gfn.samplers import LocalSearchSampler, StackLocalSearchSampler
import faulthandler
faulthandler.enable()
import ctypes
import os

# Get absolute path to the library based on this script's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust this relative path if your script is in a different folder depth
LIB_PATH = os.path.join(CURRENT_DIR, "../../src/librnaeval.so") 
LIB_PATH = os.path.normpath(LIB_PATH)

print(f"Loading library from: {LIB_PATH}")

try:
    lib = ctypes.CDLL(LIB_PATH)
    
    # --- 既存の eval_energy の定義 ---
    lib.eval_energy.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_short)]
    lib.eval_energy.restype = ctypes.c_float

    # --- 【ここを追加してください！】 pf_eval_energy の定義 ---
    # これがないと float が正しく受け取れません
    lib.pf_eval_energy.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.pf_eval_energy.restype = ctypes.c_float
    
    LIBRARY_AVAILABLE = True
except OSError as e:
    print(f"Error loading library at {LIB_PATH}: {e}")
    # Fallback or exit depending on preference. Here we exit to prevent wrong calculations.
    exit(1)

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


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # seq = "GGAAGGAGGAACCUCCUCC"
    # seq = "UGGGAAAACCC"
    # seq = "GGUCGGAAAUUCCCGGAUUUGGAACCUCGG"
    # seq = "GGGAAAUCCCGGAUUUCCCGGAUUAACCGG"
    # seq = "AACUAAAACAAUUUUUGAAGAACAGUUUCUGUACUUCAUUGGUAUGUAGAGACUUC"
    # seq = "UGCGAUCCGCGCUUCGGA"
    # seq = "GCGGCACCGUCCGCUCAAACAAACGG"
    seq = args.sequence
    seq_name = args.name
    print(f"Running: {seq_name}")
    print("Seq: ", seq)
    
    valid_pairs = enumerate_base_pairs(seq)
    # valid_pairs = filter_unstackable_pairs(valid_pairs)   #スタックを形成しうるペアに限定
    stacks = enumerate_stacks(valid_pairs)

    print("valid pairs: ", valid_pairs)
    print("stack:", stacks)

    fc = RNA.fold_compound(seq)
    G_vi = fc.pf()
    # --- pf_eval_energy の呼び出し ---
    # 1. sequence: 配列データ
    c_seq = ctypes.create_string_buffer(seq.encode('utf-8'))
    
    # 2. structure: 制約なしを表すドット列で初期化（長さ+null文字分確保される）
    # ViennaRNAのpf_foldはここに書き込みを行う可能性があるため、必ず create_string_buffer を使う
    c_structure = ctypes.create_string_buffer(b'.' * len(seq))
    
    # 3. 計算実行
    G = lib.pf_eval_energy(c_seq, c_structure)
    k = 1.9872036e-3
    T = 310.15
    RT = k * T
    if isinstance(G, list):
        G = G[1]

    if isinstance(G_vi, list):
        G_vi = G_vi[1]

    logZ_vi = -G_vi / RT
    Z_vi = np.exp(logZ_vi)
    print("Partition function Z from ViennaRNA: ", Z_vi)

    logZ = -G / RT
    ref_logZ_value = float(logZ) # すでに計算済みの変数
    Z = np.exp(logZ)
    print("Partition function Z from C library: ", Z)
    logZ = np.log(Z)

    logZ_val = float(np.log(Z))
    logZ_tensor = torch.tensor(logZ_val, dtype=torch.float)
    logZ_param = nn.Parameter(logZ_tensor)
    # logZ_module = LogZModule(logZ_param=logZ)

    min_epsilon = 0.1
    decay_factor = 0.999

    scheduler = EpsilonExponentialScheduler(args.epsilon, min_epsilon, decay_factor)

    # env = RNAPairingEnv(seq, valid_pairs, device=device)   #スタックを形成しうるペアに限定
    env = RNAPseudoknotStackingEnv(seq, stacks, device=device)
    preprocessor = KHotPreprocessor(height=len(seq)+1, ndim=len(seq))

    module_PF = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions, activation_fn="leaky_relu")
    if not args.uniform_pb:
        module_PB = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1, trunk=module_PF.trunk)
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True)

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=logZ_param)

    sampler = LocalSearchSampler(pf_estimator=pf_estimator, pb_estimator=pb_estimator)
    # sampler = LocalSearchSampler(pf_estimator=pf_estimator, pb_estimator=pb_estimator)
    gflownet = gflownet.to(device)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": args.lr_logz})

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    
    # ref_logZ_value は float型であることを確認（tensorとの比較用）
    print(f"Lower bound for logZ set to: {ref_logZ_value}")

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        epsilon = scheduler.get_epsilon()

        # 現在の logZ の値を取得
        current_logZ = gflownet.logZ.item()

        if it % 50 == 0:
            print(f"Step {it}: Eps={epsilon:.4f}, Loss={loss.item() if 'loss' in locals() else 0:.4f}, "
                  f"LogZ={current_logZ:.4f} (Ref={ref_logZ_value:.4f})")

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

        # -------------------------------------------------------------------
        # 【追加】 logZ の下限制約 (Lower Bound Constraint)
        # -------------------------------------------------------------------
        # optimizer.step() で logZ が更新された後、ref_logZ_value を下回らないように修正します。
        with torch.no_grad():
            gflownet.logZ.clamp_(min=ref_logZ_value)
        # -------------------------------------------------------------------

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            # clamp後の値を表示したい場合はここで再取得しても良いですが、
            # 基本的に current_logZ (更新前) または直近の値で推移を確認できます
            "logZ": f"{gflownet.logZ.item():.4f}", 
            "Ref": f"{ref_logZ_value:.4f}"
        })

    # --- Sampling and counting structures ---
    n_samples = 10000
    num_batches = (n_samples + args.batch_size - 1) // args.batch_size
    structures = []
    energies = []
    pair_indices = []

    seq_b = seq.encode('utf-8')

    for it in range(num_batches):
        current_batch_size = min(args.batch_size, n_samples - len(structures))
        if current_batch_size <= 0:
            break
        sampled_trajectories = sampler.sample_trajectories(env, n=args.batch_size, save_logprobs=False, save_estimator_outputs=False, epsilon=0.0)

        states = sampled_trajectories.terminating_states.tensor
        for state in states:
            # print("state: ", state)
            # 1. Convert Tensor to List
            state_list = state.cpu().tolist()
            
            # 2. Add Sentinel -1 for C function
            # Note: state_list already contains pair indices (0 for unpaired, index+1 for paired)
            pair_table = state_list + [-1]
            
            # 3. Create C array
            c_pair_table = (ctypes.c_short * len(pair_table))(*pair_table)
            
            # 4. Calculate Energy using custom library (★ Modified Part)
            try:
                energy_val = lib.eval_energy(seq_b, c_pair_table)
            except Exception:
                energy_val = 100.0 # High penalty on error
            
            # 5. Get visual string for display
            structure_str = env.state_tensor_to_dotbracket(state)
            
            structures.append(structure_str)
            energies.append(np.exp(-energy_val / RT))
            pair_indices.append(state_list)

    structure_counts = Counter(structures)
    structure_energy_dict = {}
    structure_pair_dict = {}
    for structure, energy, pair_idx in zip(structures, energies, pair_indices):
        if structure not in structure_energy_dict:
            structure_energy_dict[structure] = []
            structure_pair_dict[structure] = pair_idx
        structure_energy_dict[structure].append(energy)

    print("\n=== Sampled Structure Frequencies ===")
    for structure, count in structure_counts.most_common():
        avg_energy = sum(structure_energy_dict[structure]) / len(structure_energy_dict[structure])
        pair_idx = structure_pair_dict[structure]
        print(f"{structure}: {count} samples, score: {avg_energy:.2f}")
        print(f" Pair indices: {pair_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the estimators' modules")
    parser.add_argument("--lr_logz", type=float, default=1e-1, help="Learning rate for the logZ parameter")
    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform backward policy")
    parser.add_argument("--n_iterations", type=int, default=4000, help="Number of iterations")
    parser.add_argument("--validation_interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--validation_samples", type=int, default=1000, help="Number of validation samples to evaluate PMF")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the sampler")
    parser.add_argument("--sequence", type=str, required=True, help="RNA sequence string")
    parser.add_argument("--name", type=str, default="output", help="Name of the sequence/output file")

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