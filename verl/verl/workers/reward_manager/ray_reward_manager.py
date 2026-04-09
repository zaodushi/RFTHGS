# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

from typing import Dict, List
from verl.workers.reward_manager.utils import reward_func_timeout_ray
import ray
from ray.exceptions import GetTimeoutError  # 用于处理超时
import uuid
import os
from pathlib import Path   
import tempfile
import json
from verl.workers.reward_manager.evo_prompt import evo_prompt

delete_queue = []
individuals = [] # List[item : struct<code: str, avg_gap: float>]
import random   
import wandb


# 给vrp_evo使用的增加新个体的函数
def add_new_individual(code: str, avg_gap: float):
    global individuals
    for item in individuals:
        if item["code"] == code:
            return
    individuals.append({"code": code, "avg_gap": avg_gap})

# 把baseline code加入individuals
# curr_dir = os.path.dirname(os.path.abspath(__file__))
# path = Path(curr_dir).parent.parent.parent.parent
# baseline_avg_score = None
# with open(os.path.join(path, 'benchmark/baseline_data.json'), 'r') as f:
#             baseline_data = json.load(f)
#             baseline_avg_score = baseline_data["score"]

# add_new_individual(evo_prompt(), baseline_avg_score)

def sample_with_replacement(
    population,
    k,
    seed
) -> List:
    if k < 0:
        raise ValueError("k must be non-negative")
    if len(population) == 0:
        raise ValueError("population cannot be empty")

    rng = random.Random(seed)        
    n = len(population)

    return [population[rng.randrange(n)] for _ in range(k)]

# 每个epoch后更新数据集所用的函数。数据集大小保持不变
def get_dataframe(size: int, epoch: int):
    global individuals
    # 从individuals中随机抽样直至有size个元素
    print(f"Current individuals size: {len(individuals)}")
    prompt = sample_with_replacement(individuals, size, epoch)
    prompt = [[
        {
            "role": "system",
            "content": "You are an expert C++ optimization engineer specializing in Vehicle Routing Problems (VRP). Your role is to carefully analyze and improve algorithms while maintaining compatibility with existing code. Always verify that your modifications comply with constraints, and thoroughly double-check your solutions."
        },
        {
            "role": "user", 
            "content": evo_prompt(item["code"])
        }
        ] for item in prompt]
    # for i in range(size):
    #     index = torch.randint(0, len(individuals), (1,)).item()
    #     prompt.append([{"role": "user", "content": evo_prompt(individuals[index]["code"])}])

    return prompt

@register("ray_rm")
class RayRewardManager:
    """The reward manager."""
                                                                                            
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", timeout_seconds=120) -> None:
        """                                                                                                                                                                                                                          
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.timeout_seconds = timeout_seconds

    # 修改原始方法，使用 Ray
    def _compute_score_parallel_with_ray(self, data_sources, solution_strs, ground_truths, extra_infos):
        scores: List[float] = [0.0] * len(solution_strs)
        extra_info_dict: Dict[str, List[float]] = {}  # Key -> list of values for the batch
        print(f"Scoring process started over {len(solution_strs)} samples, waiting for results...")

        futures = []
        uuids = []
        for i in range(len(solution_strs)):
            ground_truth = ground_truths[i]
            solution_str = solution_strs[i]
            data_source = data_sources[i]
            extra_info = extra_infos[i]
            pyvrp_id = str(uuid.uuid4())
            uuids.append(pyvrp_id)

            # 提交任务给 Ray
            future = reward_func_timeout_ray.remote(
                self.compute_score, self.timeout_seconds, data_source, solution_str, ground_truth, pyvrp_id=pyvrp_id, extra_info=extra_info
            )
            futures.append(future)

        default_fail_score = {"score": -1.0, "extra_info": {"is_filter": 1}}  # Default on error which should be filtered

        # 获取任务结果，处理超时逻辑
        for i, future in enumerate(futures):
            try:
                # 设置结果返回的超时时间。与 ProcessPoolExecutor 不同，Ray 在这里通过 ray.get 的 timeout 参数控制
                task_result = ray.get(future, timeout=self.timeout_seconds)

                # 标准化 task_result 的格式
                if isinstance(task_result, dict):
                    assert 'extra_info' in task_result, f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                    score_result = task_result
                    # 如果计算结果未过滤，确保正确标记
                    if "is_filter" not in task_result["extra_info"]:
                        score_result["extra_info"].update({"is_filter": 0})
                elif isinstance(task_result, (int, float)):  # 处理标量返回结果
                    score_result = {"score": float(task_result), "extra_info": {"is_filter": 0}}
                else:
                    print(f"Unexpected task_result type for item {i}: {type(task_result)}. Using default score. Result: {task_result}")
                    print(f"Solution string: {solution_strs[i]}")
                    ray.cancel(future, force=True)
                    score_result = default_fail_score
            except GetTimeoutError:
                print(f"Timeout processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'). Using default score.")
                score_result = default_fail_score
            except Exception as e:
                print(f"Error processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'): {e}")
                import traceback
                traceback.print_exc()
                ray.cancel(future, force=True)
                score_result = default_fail_score

            # 存储最终得分
            scores[i] = float(score_result.get('score', 0.0))  # 确保 score 是 float 类型

            # 如果存在 extra_info，收集它
            if 'extra_info' in score_result and isinstance(score_result['extra_info'], dict):
                for key, value in score_result['extra_info'].items():
                    if key not in extra_info_dict:
                        # 初始化列表（例如默认值 0.0）以匹配所有项
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value
        wandb.log({
                "score/hist": wandb.Histogram(scores),
            })
        # 打印第一个数据的solution_str和score
        print(f"First sample:\n solution_str='{solution_strs[0]}'\n score={scores[0]}")
        global delete_queue
        delete_queue.extend(uuids)
        # 删除所有benchmark/pyvrp_{uuid}
        for pyvrp_id in delete_queue:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            path = Path(curr_dir).parent.parent.parent.parent
            target_dir = str(path / "benchmark" / f"pyvrp_{pyvrp_id}")
            os.system(f"rm -rf {target_dir} > /dev/null 2>&1")

            if not os.path.exists(target_dir):
                if pyvrp_id in delete_queue:
                    delete_queue.remove(pyvrp_id)

        return scores, extra_info_dict

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # already_print_data_sources = {}

        response_ids = data.batch['responses']
        sequences_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_infos = [data_item.non_tensor_batch.get('extra_info', None) for data_item in data]


        assert len(sequences_strs) == len(ground_truths) == len(data_sources)

        scores, extra_info_dict = self._compute_score_parallel_with_ray(data_sources, sequences_strs, ground_truths, extra_infos)

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        for i in range(len(data)):
            # data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


