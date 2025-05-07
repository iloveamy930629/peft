# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List, Literal, Optional

import torch


def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqeezing in the remaining dimenions.

    Args:
        task_tensors (`torch.Tensor`): The tensors that will be used to reshape `weights`.
        weights (`torch.Tensor`): The tensor to be reshaped.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


def magnitude_based_pruning(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The tensor with the pruned weights.
    """
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """
    Prune random values based on the specified fraction `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(
    tensor: torch.Tensor, density: float, method: Literal["magnitude", "random"], rescale: bool = False
) -> torch.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    if density >= 1:
        warnings.warn(f"The density {density} is greater than or equal to 1, no pruning will be performed.")
        return tensor
    elif density < 0:
        raise ValueError(f"Density should be >= 0, got {density}")
    if method == "magnitude":
        return magnitude_based_pruning(tensor, density)
    elif method == "random":
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f"Unknown method {method}")


def calculate_majority_sign_mask(
    tensor: torch.Tensor, method: Literal["total", "frequency"] = "total"
) -> torch.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`torch.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The majority sign mask.
    """

    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    """
    Merge the task tensors using disjoint merge.

    Args:
        task_tensors (`torch.Tensor`):The task tensors to merge.
        majority_sign_mask (`torch.Tensor`):The mask of the majority sign across the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)

#### Todo: modify steps of merging algorithms or add new methods in merge_utils.py ####

def task_arithmetic(task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def magnitude_prune(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`): The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


# def ties(
#     task_tensors: List[torch.Tensor],
#     weights: torch.Tensor,
#     density: float,
#     majority_sign_method: Literal["total", "frequency"] = "total",
# ) -> torch.Tensor:
#     """
#     Merge the task tensors using `ties`.

#     Args:
#         task_tensors(`List[torch.Tensor]`):The task tensors to merge.
#         weights (`torch.Tensor`):The weights of the task tensors.
#         density (`float`):The fraction of values to preserve. Should be in [0,1].
#         majority_sign_method (`str`):
#             The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

#     Returns:
#         `torch.Tensor`: The merged tensor.
#     """
#     # sparsify
#     task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
#     task_tensors = torch.stack(task_tensors, dim=0)
#     # Elect Sign
#     majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
#     # weighted task tensors
#     weights = reshape_weight_task_tensors(task_tensors, weights)
#     weighted_task_tensors = task_tensors * weights
#     # Disjoint Merge
#     mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
#     return mixed_task_tensors


def ties(
    task_tensors: List[torch.Tensor],              # τ_t：每個 LoRA delta tensor
    weights: torch.Tensor,                         # α_t：任務的加權係數
    density: float,                                # k：保留的密度比例（0~1）
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    Implements TIES-Merging Algorithm (Ilharco et al. 2022), with 3 stages:
    1. Trim (sparsify each τ_t)
    2. Elect Sign (final γ_m)
    3. Disjoint Merge (合併相容符號的 task tensor)

    Returns:
        merged_tensor: torch.Tensor  (τ_m)
    """

    num_tasks = len(task_tensors)
    shape = task_tensors[0].shape
    device = task_tensors[0].device

    # Step 1: Trim redundant parameters (個別 task 剪枝)
    trimmed_tensors = []      # τ̂_t
    signs_list = []           # γ̂_t

    for t in range(num_tasks):
        τ_t = task_tensors[t]
        τ_t = weights[t] * τ_t  # 套用任務加權係數 α_t

        # ➤ 對每個 tensor 取絕對值並找出 top-k 個位置 (按 density 定義的 k%)
        k = int(density * τ_t.numel())
        flat = τ_t.abs().view(-1)
        if k == 0:
            mask = torch.zeros_like(flat, dtype=torch.bool)
        else:
            threshold = torch.topk(flat, k)[0][-1]
            mask = (flat >= threshold)

        # ➤ 用 mask 保留重要位置，其餘設為 0
        τ_hat = torch.zeros_like(flat)
        τ_hat[mask] = flat[mask] * torch.sign(τ_t.view(-1))[mask]
        τ_hat = τ_hat.view(shape)
        trimmed_tensors.append(τ_hat)

        # ➤ 儲存 sign(τ̂_t)
        signs_list.append(torch.sign(τ_hat))

    trimmed_stack = torch.stack(trimmed_tensors, dim=0)  # shape: (n, ...)
    signs_stack = torch.stack(signs_list, dim=0)         # shape: (n, ...)

    # Step 2: Elect Final Sign γ_m
    if majority_sign_method == "total":
        γ_m = torch.sign(trimmed_stack.sum(dim=0))  # sum vote
    elif majority_sign_method == "frequency":
        # frequency vote: 多數決
        signs_sum = signs_stack.sum(dim=0)
        γ_m = torch.sign(signs_sum)
    else:
        raise ValueError("Invalid majority_sign_method")

    # Step 3: Disjoint Merge (只合併 γ̂_t 和 γ_m 一致的元素)
    merged_tensor = torch.zeros_like(trimmed_stack[0])

    for t in range(num_tasks):
        γ_hat_t = signs_list[t]
        τ_hat_t = trimmed_tensors[t]

        # mask 出與最終 sign γ_m 相同的位置
        agree_mask = (γ_hat_t == γ_m).float()

        # 加總相容 task tensor 的值
        merged_tensor += τ_hat_t * agree_mask

    # 平均化（除以每個位置的支持數）
    agree_count = (signs_stack == γ_m).sum(dim=0).clamp(min=1)  # 防止除以 0
    merged_tensor /= agree_count

    return merged_tensor


# def ties_global_trim(
#     task_tensors: List[torch.Tensor],
#     weights: torch.Tensor,
#     density: float,
#     majority_sign_method: Literal["total", "frequency"] = "total",
# ) -> torch.Tensor:
#     """
#     Variant of TIES-Merging with Global Trim:
#     1. Weighted Sum → Global Top-K Prune
#     2. Elect Majority Sign
#     3. Disjoint Merge
#     """

#     num_tasks = len(task_tensors)
#     device = task_tensors[0].device
#     shape = task_tensors[0].shape

#     # Step 1: Weighted Sum of Task Vectors
#     weighted_tensors = [w * t for w, t in zip(weights, task_tensors)]
#     sum_tensor = sum(weighted_tensors)

#     # Global Trim (on sum_tensor)
#     k = int(density * sum_tensor.numel())
#     flat = sum_tensor.abs().view(-1)
#     threshold = torch.topk(flat, k)[0][-1]
#     global_mask = (flat >= threshold).view(shape).float()

#     # Step 2: Elect Sign (γ_m)
#     stacked_signs = torch.stack([torch.sign(w * t) for w, t in zip(weights, task_tensors)], dim=0)
#     if majority_sign_method == "total":
#         γ_m = torch.sign(torch.sum(stacked_signs, dim=0))
#     elif majority_sign_method == "frequency":
#         γ_m = torch.sign(torch.sum(stacked_signs, dim=0))
#     else:
#         raise ValueError("Invalid majority_sign_method")

#     # Step 3: Disjoint Merge
#     merged_tensor = torch.zeros_like(task_tensors[0])
#     for w, t in zip(weights, task_tensors):
#         sign_t = torch.sign(w * t)
#         agree_mask = (sign_t == γ_m).float()
#         merged_tensor += w * t * agree_mask

#     agree_count = (stacked_signs == γ_m).sum(dim=0).clamp(min=1)
#     merged_tensor /= agree_count
#     merged_tensor *= global_mask

#     return merged_tensor


def ties_global_trim(
    task_tensors: List[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency", "sign"] = "total",
) -> torch.Tensor:
    """
    Variant of TIES-Merging with Global Trim:
    1. Weighted Sum → Global Top-K Prune
    2. Elect Majority Sign
    3. Disjoint Merge
    """

    num_tasks = len(task_tensors)
    device = task_tensors[0].device
    shape = task_tensors[0].shape

    # Step 1: Weighted Sum of Task Vectors
    weighted_tensors = [w * t for w, t in zip(weights, task_tensors)]
    sum_tensor = sum(weighted_tensors)

    # Global Trim (on sum_tensor)
    k = int(density * sum_tensor.numel())
    flat = sum_tensor.abs().view(-1)
    threshold = torch.topk(flat, k)[0][-1]
    global_mask = (flat >= threshold).view(shape).float()

    # Step 2: Elect Sign (γ_m)
    stacked_signs = torch.stack([torch.sign(w * t) for w, t in zip(weights, task_tensors)], dim=0)

    if majority_sign_method == "total":
        # Weighted sum of sign vectors (actually unweighted due to sign(), but follows original logic)
        γ_m = torch.sign(torch.sum(stacked_signs, dim=0))
    elif majority_sign_method == "frequency":
        # Use frequency count of +1 and -1 (majority voting)
        pos_counts = (stacked_signs > 0).sum(dim=0)
        neg_counts = (stacked_signs < 0).sum(dim=0)
        γ_m = torch.where(pos_counts >= neg_counts,
                          torch.tensor(1.0, device=device),
                          torch.tensor(-1.0, device=device))
    elif majority_sign_method == "sign":
        # Same as frequency – for clarity, they can point to same logic
        pos_counts = (stacked_signs > 0).sum(dim=0)
        neg_counts = (stacked_signs < 0).sum(dim=0)
        γ_m = torch.where(pos_counts >= neg_counts,
                          torch.tensor(1.0, device=device),
                          torch.tensor(-1.0, device=device))
    else:
        raise ValueError("Invalid majority_sign_method")

    # Step 3: Disjoint Merge
    merged_tensor = torch.zeros_like(task_tensors[0])
    for w, t in zip(weights, task_tensors):
        sign_t = torch.sign(w * t)
        agree_mask = (sign_t == γ_m).float()
        merged_tensor += w * t * agree_mask

    agree_count = (stacked_signs == γ_m).sum(dim=0).clamp(min=1)
    merged_tensor /= agree_count
    merged_tensor *= global_mask

    return merged_tensor


#### Todo: Add new methods, reuse modules in other algorithms ####
#### e.g. if you want to implement “sce” algorithm ####
"""
def sce(task_tensors: List[torch.Tensor],
    density: float = 1.0,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    # derive task vectors
    
    # S: select top-k variance elements in matrices (among different task vectors) v.s. TIES (pruning individually)
    # C: sum of squares of elements to obtain merging coefficient for each target LLM
    # E: filter elements with minority directions

    return 
"""

def sce(task_tensors, weights, density, majority_sign_method="total"):

    # stack: [num_tasks, ..., tensor_shape]
    stacked = torch.stack(task_tensors, dim=0)

    # 1. compute variance across task dimension
    variances = torch.var(stacked, dim=0)  # shape: same as tensor

    # 2. flatten and get top-d indices
    num_elements = variances.numel()
    k = int(density * num_elements)
    topk_vals, topk_indices = torch.topk(variances.view(-1), k)
    mask = torch.zeros_like(variances.view(-1))
    mask[topk_indices] = 1.0
    mask = mask.view(variances.shape)

    # 3. weighted sum of task tensors
    weighted_sum = sum(w * t for w, t in zip(weights, task_tensors))

    # 4. sign majority voting: erase conflicting elements
    stacked_signs = torch.sign(stacked)
    if majority_sign_method == "total":
        majority = torch.sign(torch.sum(stacked_signs * weights.view(-1, 1, 1), dim=0))
    elif majority_sign_method == "frequency":
        majority = torch.sign(torch.sum(stacked_signs, dim=0))
    else:
        raise ValueError("Unknown majority_sign_method")

    final_tensor = weighted_sum * (torch.sign(weighted_sum) == majority).float()
    final_tensor = final_tensor * mask  # apply pruning mask

    return final_tensor


# def sce_soft_merge(
#     task_tensors: List[torch.Tensor],
#     density: float = 0.1,
#     soft_fn: str = "sigmoid",  # or "softmax"
#     temp: float = 1.0,         # 溫度參數，用來控制 soft mask 的平滑程度
# ) -> torch.Tensor:
#     """
#     SCE with soft merge: no hard erase, no binary salient mask.
#     Elements are softly weighted based on global importance and task-wise direction alignment.

#     Args:
#         task_tensors (List[Tensor]): LoRA deltas from each task.
#         density (float): Controls how steep the importance weighting is.
#         soft_fn (str): "sigmoid" or "softmax" for importance weighting.
#         temp (float): Temperature scaling for soft weighting.

#     Returns:
#         torch.Tensor: Merged delta tensor.
#     """

#     K = len(task_tensors)
#     shape = task_tensors[0].shape
#     device = task_tensors[0].device

#     delta_stack = torch.stack(task_tensors, dim=0)          # shape: [K, ...]
#     squared_stack = delta_stack ** 2                        # shape: [K, ...]

#     # Step 1: Compute global saliency score
#     global_score = squared_stack.mean(dim=0)                # shape: [...]

#     if soft_fn == "sigmoid":
#         # Normalize and apply sigmoid as soft mask
#         normed = (global_score - global_score.mean()) / (global_score.std() + 1e-8)
#         soft_mask = torch.sigmoid(normed / temp)            # range ~ 0 to 1
#     elif soft_fn == "softmax":
#         soft_mask = torch.softmax(global_score.view(-1) / temp, dim=0).view(shape)  # normalized importance
#     else:
#         raise ValueError("soft_fn must be 'sigmoid' or 'softmax'")

#     # Step 2: Compute task-level coefficients η_j (like original SCE)
#     weighted_importance = (squared_stack * soft_mask).sum(dim=list(range(1, squared_stack.dim())))  # shape: [K]
#     eta = weighted_importance / (weighted_importance.sum() + 1e-8)                                   # shape: [K]

#     # Step 3: Soft alignment weighting by agreement with majority sign
#     mean_sign = torch.sign(delta_stack.sum(dim=0))         # shape: [...]
#     aligned_stack = []

#     for j in range(K):
#         # agreement strength: +1 if aligned, -1 if not
#         align_factor = (torch.sign(delta_stack[j]) == mean_sign).float() * 2 - 1  # in {+1, -1}
#         aligned_tensor = delta_stack[j] * (1 + align_factor) / 2  # soft keep aligned, zero otherwise
#         aligned_stack.append(aligned_tensor)

#     aligned_stack = torch.stack(aligned_stack, dim=0)      # shape: [K, ...]

#     # Step 4: Apply soft mask + η weighted sum
#     eta = eta.view(-1, *([1] * (aligned_stack.dim() - 1)))  # shape: [K, 1, 1, ...]
#     weighted_sum = (eta * aligned_stack).sum(dim=0)
#     final_tensor = weighted_sum * soft_mask                # softly scaled output

#     return final_tensor


def dare_linear(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    """
    Merge the task tensors using `dare linear`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def dare_ties(
    task_tensors: List[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    Merge the task tensors using `dare ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors
