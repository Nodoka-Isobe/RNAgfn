"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

from typing import cast, List

import torch
import torch.nn as nn
import numpy as np
import random

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gym import RNAStackingEnv
from gfn.estimators import Estimator, ScalarEstimator
from gfn.gflownet.base import TrajectoryBasedGFlowNet, loss_reduce
from gfn.utils.handlers import (
    is_callable_exception_handler,
    warn_about_recalculating_logprobs,
)


class TBGFlowNet(TrajectoryBasedGFlowNet):
    r"""GFlowNet for the Trajectory Balance loss.

    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the
    DAG. $\mathcal{O}_3$ is the set of backward probability functions consistent with
    the DAG, or a singleton thereof, if self.pb is a fixed DiscretePBEstimator.

    See [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)
    for more details.

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
            pb is therefore always 1.
        logZ: A learnable parameter or a ScalarEstimator instance (for conditional GFNs).
        constant_pb: Whether to ignore pb e.g., the GFlowNet DAG is a tree, and pb
            is therefore always 1. Must be set explicitly by user to ensure that pb
            is an Estimator except under this special case.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logZ: nn.Parameter | ScalarEstimator | None = None,
        temparature: float = 310.15,
        init_logZ: float = 0.0,
        constant_pb: bool = False,
        log_reward_clip_min: float = -float("inf"),
    ):
        """Initializes a TBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
                pb is therefore always 1.
            logZ: A learnable parameter or a ScalarEstimator instance (for
                conditional GFNs).
            init_logZ: The initial value for the logZ parameter (used if logZ is None).
            constant_pb: Whether to ignore pb e.g., the GFlowNet DAG is a tree, and pb
                is therefore always 1. Must be set explicitly by user to ensure that pb
                is an Estimator except under this special case.
            log_reward_clip_min: If finite, clips log rewards to this value.
        """
        super().__init__(
            pf, pb, constant_pb=constant_pb, log_reward_clip_min=log_reward_clip_min
        )

        self.logZ = logZ or nn.Parameter(torch.tensor(init_logZ))

    def logz_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of named parameters containing 'logZ' in their name.

        Returns:
            A dictionary of named parameters containing 'logZ' in their name.
        """
        return {k: v for k, v in dict(self.named_parameters()).items() if "logZ" in k}

    def logz_parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters containing 'logZ' in their name.

        Returns:
            A list of parameters containing 'logZ' in their name.
        """
        return [v for k, v in dict(self.named_parameters()).items() if "logZ" in k]

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the trajectory balance loss.

        The trajectory balance loss is described in section 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259).

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed trajectory balance loss as a tensor. The shape depends on the
            reduction method.
        """
        del env  # unused

        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)

        scores = self.get_scores(
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )

        # If the conditions values exist, we pass them to self.logZ
        # (should be a ScalarEstimator or equivalent).
        if trajectories.conditions is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                assert isinstance(self.logZ, ScalarEstimator)
                logZ = self.logZ(trajectories.conditions)
        else:
            logZ = self.logZ

        logZ = cast(torch.Tensor, logZ)

        # scores (log_pf - log_pb - log_r) を安全な範囲に絞る
        scores = torch.clamp(scores, min=-20.0, max=20.0)

        scores = (scores + logZ.squeeze()).pow(2)

        loss = loss_reduce(scores, reduction)

        if torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss


class LogPartitionVarianceGFlowNet(TrajectoryBasedGFlowNet):
    """GFlowNet for the Log Partition Variance loss.

    The log partition variance loss is described in section 3.2 of
    [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446).

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator.
        constant_pb: Whether to ignore pb e.g., the GFlowNet DAG is a tree, and pb
            is therefore always 1. Must be set explicitly by user to ensure that pb
            is an Estimator except under this special case.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the log partition variance loss.

        The log partition variance loss is described in section 3.2 of
        [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446).

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed log partition variance loss as a tensor. The shape depends on
            the reduction method.
        """
        del env  # unused

        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)

        scores = self.get_scores(
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )

        scores = (scores - scores.mean()).pow(2)

        loss = loss_reduce(scores, reduction)

        if torch.isnan(loss).any():
            raise ValueError("loss is NaN.")

        return loss


class ReplicaGFlowNet(nn.Module):
    def __init__(
        self,
        replicas: List[TBGFlowNet],
        temperatures: List[float],
    ):
        super().__init__()

        self.replicas = nn.ModuleList(replicas)
        self.temperatures = temperatures
        self.num_replicas = len(temperatures)
        self.gas_constant = 1.9872036e-3

    def compute_all_losses(
        self,
        env: RNAStackingEnv,
        replica_trajectories: List[Trajectories],
        reduction: str = "mean",
        do_exchange: bool = False,
    ) -> List[torch.Tensor]:

        # 1. レプリカ交換の実行
        if do_exchange:
            replica_trajectories = self.perform_replica_exchange(
                env,
                replica_trajectories,
            )

        # 2. 損失計算
        total_loss = 0
        losses = []

        for i, (replica, trajectories) in enumerate(
            zip(self.replicas, replica_trajectories)
        ):
            # 環境の温度を現在のレプリカの温度に設定
            env.temperature = self.temperatures[i]

            log_rewards = torch.log(
                env.reward(trajectories.terminating_states)
            )

            trajectories._log_rewards = log_rewards

            l = replica.loss(
                env,
                trajectories,
                reduction=reduction,
            )

            losses.append(l)
            total_loss += l

        return total_loss, losses

    def perform_replica_exchange(
        self,
        env,
        trajectories_list: List[Trajectories],
    ):
        """
        隣接するレプリカ間で軌道（構造）の交換を行う
        """
        n_rep = self.num_replicas

        betas = [
            1.0 / (self.gas_constant * T)
            for T in self.temperatures
        ]

        energies = []

        for trajs in trajectories_list:
            structs = [
                env.state_tensor_to_dotbracket(s)
                for s in trajs.terminating_states
            ]

            e = np.mean([
                env.fc.eval_structure(s)
                for s in structs
            ])

            energies.append(e)

        start_idx = random.choice([0, 1])

        for rr in range(start_idx, n_rep - 1, 2):
            beta_curr, beta_next = betas[rr], betas[rr + 1]
            energy_curr, energy_next = energies[rr], energies[rr + 1]

            delta_beta = beta_next - beta_curr
            delta_energy = energy_next - energy_curr

            w = np.exp(delta_beta * delta_energy)

            if np.random.uniform(0, 1) < w:
                # print("replica exchange ", rr, "and ", rr+1)

                # 軌道データの入れ替え
                trajectories_list[rr], trajectories_list[rr + 1] = (
                    trajectories_list[rr + 1],
                    trajectories_list[rr],
                )

                # エネルギー値の入れ替え
                energies[rr], energies[rr + 1] = (
                    energies[rr + 1],
                    energies[rr],
                )

        return trajectories_list

# class ReplicaGFlowNet_one(nn.Module):
#     def __init__(
#         self,
#         replicas: List[nn.Module], # TBGFlowNet のリスト
#         temperatures: List[float]
#     ):
#         super().__init__()
#         self.replicas = nn.ModuleList(replicas)
#         self.temperatures = temperatures
#         self.num_replicas = len(temperatures)
#         self.gas_constant = 1.9872036e-3

#     def compute_all_losses(
#         self,
#         env: RNAStackingEnv,
#         replica_trajectories: List[Trajectories],
#         reduction: str = "mean",
#         do_exchange: bool = True
#     ) -> tuple:
        
#         # 交換前の、各レプリカの「オリジナルの所属ラベル」をバッチサイズ分作成
#         batch_size = replica_trajectories[0].n_trajectories
#         device = replica_trajectories[0].states.device
#         replica_origin_map = [
#             torch.full((batch_size,), idx, dtype=torch.long, device=device) 
#             for idx in range(self.num_replicas)
#         ]

#         # 1. レプリカ交換の実行（内部で origin_map も一緒にスワップ）
#         if do_exchange:
#             replica_trajectories, replica_origin_map = self.perform_replica_exchange(env, replica_trajectories, replica_origin_map)

#         # 2. 損失計算 ＆ 使用サンプルの全出力
#         total_loss = 0
#         losses = []
        
#         print(f"\n=================== LEARNING SAMPLE TRACKING ===================")
#         for i, (replica, trajectories) in enumerate(zip(self.replicas, replica_trajectories)):
            
#             # --- マスクの再同期 ---
#             # バッファの混入がないため、純粋な交換後データに対してマスクを適用
#             env.update_masks(trajectories.states)

#             # 環境の温度を現在のレプリカの温度に設定
#             env.temperature = self.temperatures[i]
           
#             # 報酬の再計算
#             log_rewards = torch.log(env.reward(trajectories.terminating_states))
#             trajectories._log_rewards = log_rewards
           
#             # --- ログ出力部分 ---
#             current_n_trajs = trajectories.n_trajectories 
            
#             print(f"\n[Learning Layer] Replica {i:02d} (Current T={self.temperatures[i]:.1f}K) is updating Policy using:")
            
#             structs = [env.state_tensor_to_dotbracket(s) for s in trajectories.terminating_states]
#             energies = [env.fc.eval_structure(s) for s in structs]
#             origins = replica_origin_map[i].cpu().tolist()
            
#             for idx in range(current_n_trajs):
#                 # 全てが今現在のイテレーションで生成されたデータ（ONLINE）になります
#                 source_tag = "[ONLINE]"
#                 orig_rep = origins[idx] if idx < len(origins) else i
#                 exchange_tag = "[EXCHANGED]" if orig_rep != i else "[LOCAL]"
                
#                 print(f"  └─ Batch {idx:02d} | {source_tag} {exchange_tag} Born in Rep {orig_rep:02d} -> Struct: {structs[idx]} (E={energies[idx]:.2f})")

#             # 実際のLoss計算と逆伝播の準備
#             l = replica.loss(env, trajectories, reduction=reduction)
#             losses.append(l)
#             total_loss += l
           
#         print(f"===================================================================\n")
#         return total_loss, losses

#     def perform_replica_exchange(self, env, trajectories_list: List[Trajectories], replica_origin_map: List[torch.Tensor]):
#         """
#         隣接するレプリカ間で軌道（構造）の交換を行い、そのサンプルの出自ラベルも追跡する（偶奇連続スキャン版）
#         """
#         n_rep = self.num_replicas
#         betas = [1.0 / (self.gas_constant * T) for T in self.temperatures]
#         batch_size = trajectories_list[0].n_trajectories
#         device = trajectories_list[0].states.device
       
#         energies_matrix = []
#         structures_matrix = []
#         for trajs in trajectories_list:
#             structs = [env.state_tensor_to_dotbracket(s) for s in trajs.terminating_states]
#             structures_matrix.append(structs)
#             e_list = [env.fc.eval_structure(s) for s in structs]
#             energies_matrix.append(torch.tensor(e_list, device=device))

#         # 偶数始まりか奇数始まりかをランダムに決定し、連続スキャン
#         start_idx = random.choice([0, 1])
       
#         for rr in range(start_idx, n_rep - 1, 2):
#             beta_curr, beta_next = betas[rr], betas[rr+1]

#             delta_beta = beta_next - beta_curr
#             delta_energy = energies_matrix[rr+1] - energies_matrix[rr]
#             low_w = delta_beta * delta_energy
#             probs = torch.exp(torch.clamp(low_w, max=0.0))

#             random_vals = torch.rand(batch_size, device=device)
#             exchange_mask = random_vals < probs

#             num_exchanges = exchange_mask.sum().item()
#             if num_exchanges > 0:
#                 print(f"\n [Exchange Event] Replica {rr} (T={self.temperatures[rr]:.1f}) <-> Replica {rr+1} (T={self.temperatures[rr+1]:.1f})")
               
#                 exchanged_indices = torch.where(exchange_mask)[0].cpu().tolist()
#                 for idx in exchanged_indices:
#                     s_curr = structures_matrix[rr][idx]
#                     e_curr = energies_matrix[rr][idx].item()
#                     s_next = structures_matrix[rr+1][idx]
#                     e_next = energies_matrix[rr+1][idx].item()

#                     print(f"  ├── Batch {idx:02d} SWAP | Rep{rr}: {s_curr} (E={e_curr:.2f}) <=> Rep{rr+1}: {s_next} (E={e_next:.2f})")

#                 # 軌道データの中身（Tensor）を個別に入れ替える
#                 self.swap_trajectories_partially(
#                     trajectories_list[rr],
#                     trajectories_list[rr+1],
#                     exchange_mask
#                 )
                
#                 # 出自ラベルテンソル(origin_map)の中身もマスクを使ってスワップ
#                 tmp_origin = replica_origin_map[rr][exchange_mask].clone()
#                 replica_origin_map[rr][exchange_mask] = replica_origin_map[rr+1][exchange_mask]
#                 replica_origin_map[rr+1][exchange_mask] = tmp_origin
               
#                 # エネルギー行列も同期
#                 tmp_e = energies_matrix[rr][exchange_mask].clone()
#                 energies_matrix[rr][exchange_mask] = energies_matrix[rr+1][exchange_mask]
#                 energies_matrix[rr+1][exchange_mask] = tmp_e
               
#         return trajectories_list, replica_origin_map

#     def swap_trajectories_partially(self, trajs1, trajs2, mask):
#         """ 長さが異なる軌道間でも安全にサンプルを入れ替える """
#         trajs1.states.tensor, trajs2.states.tensor = self.pad_tensors(
#             trajs1.states.tensor, trajs2.states.tensor, value=-1
#         )
#         trajs1.actions.tensor, trajs2.actions.tensor = self.pad_tensors(
#             trajs1.actions.tensor, trajs2.actions.tensor, value=-1
#         )

#         # States & Actions のスワップ
#         tmp_s = trajs1.states.tensor[:, mask].clone()
#         trajs1.states.tensor[:, mask] = trajs2.states.tensor[:, mask]
#         trajs2.states.tensor[:, mask] = tmp_s

#         tmp_a = trajs1.actions.tensor[:, mask].clone()
#         trajs1.actions.tensor[:, mask] = trajs2.actions.tensor[:, mask]
#         trajs2.actions.tensor[:, mask] = tmp_a

#         # 終端インデックスと報酬のスワップ
#         tmp_idx = trajs1.terminating_idx[mask].clone()
#         trajs1.terminating_idx[mask] = trajs2.terminating_idx[mask]
#         trajs2.terminating_idx[mask] = tmp_idx

#         if trajs1._log_rewards is not None:
#             tmp_r = trajs1._log_rewards[mask].clone()
#             trajs1._log_rewards[mask] = trajs2._log_rewards[mask]
#             trajs2._log_rewards[mask] = tmp_r

#     def pad_tensors(self, a, b, value):
#         if a.shape[0] == b.shape[0]:
#             return a, b
#         max_len = max(a.shape[0], b.shape[0])
#         def pad(t, target_len):
#             pad_size = target_len - t.shape[0]
#             p = torch.full((pad_size, *t.shape[1:]), value, dtype=t.dtype, device=t.device)
#             return torch.cat([t, p], dim=0)
#         return pad(a, max_len), pad(b, max_len)

class ReplicaGFlowNet_one(nn.Module):
    def __init__(self, replicas: List[nn.Module], temperatures: List[float]):
        super().__init__()
        self.replicas = nn.ModuleList(replicas)
        self.temperatures = temperatures
        self.num_replicas = len(temperatures)
        self.gas_constant = 1.9872036e-3
        
        # ─── 【追加】軌跡追跡用のロギングバッファ ───
        self.exchange_count = 0
        # 各初期サンプル(インデックス)が、どの「逆温度」にいるかを記録するリスト
        # history[サンプルID] = [beta_t0, beta_t1, ...]
        self.trajectory_history = None 
        # サンプルが今どのレプリカ（温度インデックス）にいるかを追跡する配列
        self.sample_current_positions = None

    def perform_replica_exchange(self, env, trajectories_list: List[Trajectories], replica_origin_map: List[torch.Tensor]):
        n_rep = self.num_replicas
        betas = [1.0 / (self.gas_constant * T) for T in self.temperatures]
        # 論文の図のように最大値を1.0に正規化したInverse T（逆温度）
        max_beta = max(betas)
        normalized_betas = [b / max_beta for b in betas]
        
        batch_size = trajectories_list[0].n_trajectories
        device = trajectories_list[0].states.device
        
        # ─── 【追加】初回ステップ時に追跡配列を初期化 ───
        if self.trajectory_history is None:
            # 全サンプル数 = レプリカ数 × バッチサイズ
            total_samples = n_rep * batch_size
            self.trajectory_history = [[] for _ in range(total_samples)]
            # 初期状態では サンプルID = レプリカID * batch_size + バッチ内インデックス
            self.sample_current_positions = np.arange(total_samples).reshape(n_rep, batch_size)
       
        energies_matrix = []
        structures_matrix = []
        for trajs in trajectories_list:
            structs = [env.state_tensor_to_dotbracket(s) for s in trajs.terminating_states]
            structures_matrix.append(structs)
            e_list = [env.fc.eval_structure(s) for s in structs]
            energies_matrix.append(torch.tensor(e_list, device=device))

        start_idx = random.choice([0, 1])
       
        for rr in range(start_idx, n_rep - 1, 2):
            beta_curr, beta_next = betas[rr], betas[rr+1]
            delta_beta = beta_next - beta_curr
            delta_energy = energies_matrix[rr+1] - energies_matrix[rr]
            low_w = delta_beta * delta_energy
            probs = torch.exp(torch.clamp(low_w, max=0.0))

            random_vals = torch.rand(batch_size, device=device)
            exchange_mask = random_vals < probs

            num_exchanges = exchange_mask.sum().item()
            if num_exchanges > 0:
                print(f"\n [Exchange Event] Replica {rr} (T={self.temperatures[rr]:.1f}) <-> Replica {rr+1} (T={self.temperatures[rr+1]:.1f})")
               
                exchanged_indices = torch.where(exchange_mask)[0].cpu().tolist()
                for idx in exchanged_indices:
                    s_curr = structures_matrix[rr][idx]
                    e_curr = energies_matrix[rr][idx].item()
                    s_next = structures_matrix[rr+1][idx]
                    e_next = energies_matrix[rr+1][idx].item()
                    print(f"  ├── Batch {idx:02d} SWAP | Rep{rr}: {s_curr} (E={e_curr:.2f}) <=> Rep{rr+1}: {s_next} (E={e_next:.2f})")

                # 軌道データと出自ラベルのスワップ（既存処理）
                self.swap_trajectories_partially(trajectories_list[rr], trajectories_list[rr+1], exchange_mask)
                
                tmp_origin = replica_origin_map[rr][exchange_mask].clone()
                replica_origin_map[rr][exchange_mask] = replica_origin_map[rr+1][exchange_mask]
                replica_origin_map[rr+1][exchange_mask] = tmp_origin
               
                tmp_e = energies_matrix[rr][exchange_mask].clone()
                energies_matrix[rr][exchange_mask] = energies_matrix[rr+1][exchange_mask]
                energies_matrix[rr+1][exchange_mask] = tmp_e

                # ─── 【追加】交換に連動して、サンプルIDの現在位置配列もスワップ ───
                mask_np = exchange_mask.cpu().numpy()
                tmp_sample_ids = self.sample_current_positions[rr, mask_np].copy()
                self.sample_current_positions[rr, mask_np] = self.sample_current_positions[rr+1, mask_np]
                self.sample_current_positions[rr+1, mask_np] = tmp_sample_ids

        # ─── 【追加】全サンプルの現在の位置（逆温度）を履歴に記録 ───
        for r_idx in range(n_rep):
            current_temp = self.temperatures[r_idx]  # 正規化せず、温度の数値をそのまま使用
            for b_idx in range(batch_size):
                sample_id = self.sample_current_positions[r_idx, b_idx]
                self.trajectory_history[sample_id].append(current_temp)
                
        self.exchange_count += 1
        return trajectories_list, replica_origin_map

class ReplicaGFlowNet_replaybuffer(nn.Module):
    def __init__(
        self,
        replicas: List[TBGFlowNet],
        temperatures: List[float],
        max_buffer_size: int = 1000,
        replay_ratio: float = 0.3
    ):
        super().__init__()
        self.replicas = nn.ModuleList(replicas)
        self.temperatures = temperatures
        self.num_replicas = len(temperatures)
        self.gas_constant = 1.9872036e-3

        self.buffers = [[] for _ in range(self.num_replicas)]
        self.max_buffer_size = max_buffer_size
        self.replay_ratio = replay_ratio

    def compute_all_losses(
        self,
        env: RNAStackingEnv,
        replica_trajectories: List[Trajectories],
        reduction: str = "mean",
        do_exchange: bool = False
    ) -> List[torch.Tensor]:
       
        # 交換前の、各レプリカの「オリジナルの所属ラベル」をバッチサイズ分作成
        # 例: replica_origin_map[3][5] = 3 (レプリカ3のバッチ内インデックス5のサンプルは、元々レプリカ3出身)
        batch_size = replica_trajectories[0].n_trajectories
        device = replica_trajectories[0].states.device
        replica_origin_map = [torch.full((batch_size,), idx, dtype=torch.long, device=device) for idx in range(self.num_replicas)]

        # 1. レプリカ交換の実行（内部で origin_map も一緒にスワップさせます）
        if do_exchange:
            replica_trajectories, replica_origin_map = self.perform_replica_exchange(env, replica_trajectories, replica_origin_map)

        for i in range(self.num_replicas):
            self.set_buffer(i, replica_trajectories[i], env)
        # 2. 損失計算 ＆ 使用サンプルの全出力
        total_loss = 0
        losses = []
        
        print(f"\n=================== LEARNING SAMPLE TRACKING ===================")
        for i, (replica, trajectories) in enumerate(zip(self.replicas, replica_trajectories)):
            # Replay bufferからサンプルを3割持ってくる(少なくとも1イテレーション分以上溜まっていたら)
            n_replay = int(trajectories.n_trajectories * self.replay_ratio)
            replay_data = self.sample_from_buffer(i, n_replay) if len(self.buffers[i]) >= trajectories.n_trajectories else None
            
            n_online = trajectories.n_trajectories
            trajectories.is_replay = torch.zeros(n_online, dtype=torch.bool, device=trajectories.states.device)

            if replay_data is not None:
                n_online = trajectories.n_trajectories - n_replay
                
                trajectories.is_replay = trajectories.is_replay[:n_online]
                
                t1_s, t2_s = self.pad_tensors(trajectories.states.tensor[:, :n_online], replay_data["states"], value=-1)
                trajectories.states.tensor = torch.cat([t1_s, t2_s], dim=1)
                
                t1_a, t2_a = self.pad_tensors(trajectories.actions.tensor[:, :n_online], replay_data["actions"], value=-1)
                trajectories.actions.tensor = torch.cat([t1_a, t2_a], dim=1)
                
                trajectories.terminating_idx = torch.cat([trajectories.terminating_idx[:n_online], replay_data["terminating_idx"]], dim=0)

                replay_flags = torch.ones(n_replay, dtype=torch.bool, device=trajectories.states.device)
                trajectories.is_replay = torch.cat([trajectories.is_replay, replay_flags], dim=0)

            # --- マスクの再同期 ---
            env.update_masks(trajectories.states)

            # 環境の温度を現在のレプリカの温度に設定
            env.temperature = self.temperatures[i]
           
            # 報酬の再計算
            log_rewards = torch.log(env.reward(trajectories.terminating_states))
            trajectories._log_rewards = log_rewards
           
           #print部分
            current_n_trajs = trajectories.n_trajectories 
            
            print(f"\n[Learning Layer] Replica {i:02d} (Current T={self.temperatures[i]:.1f}K) is updating Policy using:")
            
            structs = [env.state_tensor_to_dotbracket(s) for s in trajectories.terminating_states]
            energies = [env.fc.eval_structure(s) for s in structs]
            origins = replica_origin_map[i].cpu().tolist()
            
            # バッファ由来フラグ（is_replay）を取得
            is_rep_flags = getattr(trajectories, 'is_replay', None)
            
            for idx in range(current_n_trajs):
                # バッファ由来（REPLAY）かどうかの判定
                is_replay_sample = (is_rep_flags is not None and idx < len(is_rep_flags) and is_rep_flags[idx].item())
                
                if is_replay_sample:
                    # REPLAYの時は 出自情報を出さず、シンプルなログにする
                    source_tag = "[REPLAY]"
                    print(f"  └─ Batch {idx:02d} | {source_tag} -> Struct: {structs[idx]} (E={energies[idx]:.2f})")
                else:
                    # ONLINEの時は 通常通り交換履歴（Born in）を出力する
                    source_tag = "[ONLINE]"
                    orig_rep = origins[idx] if idx < len(origins) else i
                    exchange_tag = "[EXCHANGED]" if orig_rep != i else "[LOCAL]"
                    
                    print(f"  └─ Batch {idx:02d} | {source_tag} {exchange_tag} Born in Rep {orig_rep:02d} -> Struct: {structs[idx]} (E={energies[idx]:.2f})")
            #print部分

            # 実際のLoss計算と逆伝播の準備
            l = replica.loss(env, trajectories, reduction=reduction)
            losses.append(l)
            total_loss += l
           
        return total_loss, losses

    def perform_replica_exchange(self, env, trajectories_list: List[Trajectories], replica_origin_map: List[torch.Tensor]):
        """
        隣接するレプリカ間で軌道（構造）の交換を行い、そのサンプルの出自ラベルも追跡する
        """
        n_rep = self.num_replicas
        betas = [1.0 / (self.gas_constant * T) for T in self.temperatures]
        batch_size = trajectories_list[0].n_trajectories
        device = trajectories_list[0].states.device
       
        energies_matrix = []
        structures_matrix = []
        for trajs in trajectories_list:
            structs = [env.state_tensor_to_dotbracket(s) for s in trajs.terminating_states]
            structures_matrix.append(structs)
            e_list = [env.fc.eval_structure(s) for s in structs]
            energies_matrix.append(torch.tensor(e_list, device=device))

        start_idx = random.choice([0, 1])
       
        for rr in range(start_idx, n_rep - 1, 2):
            beta_curr, beta_next = betas[rr], betas[rr+1]

            delta_beta = beta_next - beta_curr
            delta_energy = energies_matrix[rr+1] - energies_matrix[rr]
            low_w = delta_beta * delta_energy
            probs = torch.exp(torch.clamp(low_w, max=0.0))

            random_vals = torch.rand(batch_size, device=device)
            exchange_mask = random_vals < probs

            num_exchanges = exchange_mask.sum().item()
            if num_exchanges > 0:
                print(f"\n [Exchange Event] Replica {rr} (T={self.temperatures[rr]:.1f}) <-> Replica {rr+1} (T={self.temperatures[rr+1]:.1f})")
               
                exchanged_indices = torch.where(exchange_mask)[0].cpu().tolist()
                for idx in exchanged_indices:
                    s_curr = structures_matrix[rr][idx]
                    e_curr = energies_matrix[rr][idx].item()
                    s_next = structures_matrix[rr+1][idx]
                    e_next = energies_matrix[rr+1][idx].item()

                    print(f"  ├── Batch {idx:02d} SWAP | Rep{rr}: {s_curr} (E={e_curr:.2f}) <=> Rep{rr+1}: {s_next} (E={e_next:.2f})")

                # 軌道データの中身（Tensor）を個別に入れ替える
                self.swap_trajectories_partially(
                    trajectories_list[rr],
                    trajectories_list[rr+1],
                    exchange_mask
                )
                
                # --- 出自ラベルテンソル(origin_map)の中身もマスクを使ってスワップ ---
                tmp_origin = replica_origin_map[rr][exchange_mask].clone()
                replica_origin_map[rr][exchange_mask] = replica_origin_map[rr+1][exchange_mask]
                replica_origin_map[rr+1][exchange_mask] = tmp_origin
               
                # エネルギー行列も更新
                tmp_e = energies_matrix[rr][exchange_mask].clone()
                energies_matrix[rr][exchange_mask] = energies_matrix[rr+1][exchange_mask]
                energies_matrix[rr+1][exchange_mask] = tmp_e
               
        return trajectories_list, replica_origin_map

    def swap_trajectories_partially(self, trajs1, trajs2, mask):
        """ 長さが異なる軌道間でも安全にサンプルを入れ替える """
        trajs1.states.tensor, trajs2.states.tensor = self.pad_tensors(
            trajs1.states.tensor, trajs2.states.tensor, value=-1
        )
        trajs1.actions.tensor, trajs2.actions.tensor = self.pad_tensors(
            trajs1.actions.tensor, trajs2.actions.tensor, value=-1
        )

        # States & Actions のスワップ
        tmp_s = trajs1.states.tensor[:, mask].clone()
        trajs1.states.tensor[:, mask] = trajs2.states.tensor[:, mask]
        trajs2.states.tensor[:, mask] = tmp_s

        tmp_a = trajs1.actions.tensor[:, mask].clone()
        trajs1.actions.tensor[:, mask] = trajs2.actions.tensor[:, mask]
        trajs2.actions.tensor[:, mask] = tmp_a

        # 終端インデックスと報酬のスワップ
        tmp_idx = trajs1.terminating_idx[mask].clone()
        trajs1.terminating_idx[mask] = trajs2.terminating_idx[mask]
        trajs2.terminating_idx[mask] = tmp_idx

        if trajs1._log_rewards is not None:
            tmp_r = trajs1._log_rewards[mask].clone()
            trajs1._log_rewards[mask] = trajs2._log_rewards[mask]
            trajs2._log_rewards[mask] = tmp_r

    def pad_tensors(self, a, b, value):
        if a.shape[0] == b.shape[0]:
            return a, b
        max_len = max(a.shape[0], b.shape[0])
        def pad(t, target_len):
            pad_size = target_len - t.shape[0]
            p = torch.full((pad_size, *t.shape[1:]), value, dtype=t.dtype, device=t.device)
            return torch.cat([t, p], dim=0)
        return pad(a, max_len), pad(b, max_len)

    def set_buffer(self, replica_idx, trajectories, env: RNAStackingEnv):
        for idx in range(trajectories.n_trajectories):
            # 1. このサンプルの時間軸全体の歴史を取り出す (time_len, state_dim)
            full_states = trajectories.states.tensor[:, idx]
            
            # 2. 🔥【バグ回避の核心】パディングの値（-1）を除外して、このサンプルの本当の「最後の有効な状態」を見つける
            # 通常、終了状態 (sf) の一歩手前、もしくは -1 が始まる直前のインデックスが終端状態です
            valid_indices = torch.where(torch.all(full_states != -1, dim=1))[0]
            
            if len(valid_indices) == 0:
                # 万が一すべて -1 だった場合の安全策（初期状態を使う）
                final_state_raw = full_states[0]
            else:
                # 有効な状態のうち、いちばん最後のステップを終端状態としてダイレクトに指定
                last_valid_step_idx = valid_indices[-1].item()
                final_state_raw = full_states[last_valid_step_idx]

            # 3. あとは安全に取り出した終端状態を使って構造文字列と報酬を計算
            struct_str = env.state_tensor_to_dotbracket(final_state_raw)
            energy = env.fc.eval_structure(struct_str)
            reward = np.exp(-energy / (self.gas_constant * self.temperatures[replica_idx]))
            
            # バッファに保存するためのパーツをクローン
            state = trajectories.states.tensor[:, idx].clone()
            action = trajectories.actions.tensor[:, idx].clone()
            term_idx = trajectories.terminating_idx[idx].clone()
            
            sample_data = {
                "state": state,
                "action": action,
                "terminating_idx": term_idx,
                "reward": reward
            }
            
            self.buffers[replica_idx].append(sample_data)
            if len(self.buffers[replica_idx]) > self.max_buffer_size:
                self.buffers[replica_idx].pop(0)

    def sample_from_buffer(self, replica_idx: int, n_samples: int) -> dict:
        """
        【自作関数】バッファから論文のR-PRS（報酬優先）でサンプルを引き出す関数
        """
        buffer = self.buffers[replica_idx]
        if len(buffer) == 0:
            return None
            
        # 論文の R-PRS: 報酬（スコア）をそのまま重みとして確率分布を作る
        rewards = np.array([sample["reward"] for sample in buffer])
        # 万が一すべて報酬ゼロだった場合の安全策
        if rewards.sum() == 0:
            probs = np.ones(len(buffer)) / len(buffer)
        else:
            probs = rewards / rewards.sum()
            
        # 確率に基づいてインデックスをランダムチョイス
        chosen_indices = np.random.choice(len(buffer), size=n_samples, p=probs, replace=True)
        states_list = []
        actions_list = []
        term_idx_list = []

        for idx in chosen_indices:
            s_candidate = buffer[idx]["state"]
            a_candidate = buffer[idx]["action"]
            t_candidate = buffer[idx]["terminating_idx"]
            
            if len(states_list) == 0:
                # 最初の1個目はそのまま登録
                states_list.append(s_candidate)
                actions_list.append(a_candidate)
            else:
                # 2個目以降は、すでにリストにある暫定テンソルと長さを揃えながら追加していく
                # states_list[0] を基準に現在の候補とパディングを合わせる
                # (これを行うことで、全サンプルの時間軸サイズが最も長いものに自動で統一されます)
                current_stacked_states = torch.stack(states_list, dim=1)
                padded_stacked, padded_candidate = self.pad_tensors(current_stacked_states, s_candidate.unsqueeze(1), value=-1)
                # 分解してリストに戻す
                states_list = [padded_stacked[:, i] for i in range(padded_stacked.shape[1])] + [padded_candidate[:, 0]]
                
                # actions も同様にパディングして揃える
                current_stacked_actions = torch.stack(actions_list, dim=1)
                padded_act_stacked, padded_act_candidate = self.pad_tensors(current_stacked_actions, a_candidate.unsqueeze(1), value=-1)
                actions_list = [padded_act_stacked[:, i] for i in range(padded_act_stacked.shape[1])] + [padded_act_candidate[:, 0]]
                
            term_idx_list.append(t_candidate)
        
        return {
            "states": torch.stack(states_list, dim=1),       # (time, batch, state_dim)
            "actions": torch.stack(actions_list, dim=1),     # (time-1, batch)
            "terminating_idx": torch.stack(term_idx_list, dim=0) # (batch,)
        }