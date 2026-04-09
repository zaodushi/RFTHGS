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
from ray.exceptions import GetTimeoutError  # For handling timeouts
import uuid
import os
from pathlib import Path   
import tempfile
import json
from verl.workers.reward_manager.evo_prompt import evo_prompt, baseline_code

from verl.workers.reward_manager.cpp_duplicate import is_duplicate_cpp_code
from verl.workers.reward_manager.shared_memory_kvstore import SharedMemoryKeyValueStore as KeyValueStore
import random   
import wandb
import traceback
import re
import numpy as np


delete_queue = []
kv_store = KeyValueStore(namespace="prompt_individuals")
if not kv_store.exists("individuals"):
    kv_store.put("individuals", [])


# Function to add new individuals for vrp_evo
def add_new_individual(code: str, avg_gap: float):
    individuals = kv_store.get("individuals")
    if individuals is None:
        print("Error: individuals is None")
        return
    prompt = evo_prompt(code)
    individuals.append({"prompt": prompt, "avg_gap": avg_gap})
    # Remove duplicates
    individuals = list({item["prompt"]: item for item in individuals}.values())
    kv_store.update("individuals", lambda x: individuals)




def normalise(txt: str) -> str:
    txt = txt.lower()                      
    txt = re.sub(r'\s+', ' ', txt)         
    return txt.strip()

def get_gap_by_prompt(prompt: str):
    individuals = kv_store.get("individuals")
    for item in individuals:
        if normalise(item["prompt"]) == normalise(prompt):
            return item["avg_gap"]
    print(f"warning: can not find prompt in individuals.") 
    return None    

def get_code_by_prompt(prompt: str):
    return prompt.split("```cpp")[-1].split("```")[0]

# Add baseline code to individuals
curr_dir = os.path.dirname(os.path.abspath(__file__))
path = Path(curr_dir).parent.parent.parent.parent
baseline_avg_score = None
with open(os.path.join(path, 'benchmark/baseline_data.json'), 'r') as f:
    baseline_data = json.load(f)
    baseline_avg_score = baseline_data['avg_gap']

add_new_individual(baseline_code, baseline_avg_score)

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

# Function to update dataset after each epoch. Dataset size remains constant
def get_dataframe(size: int, epoch: int):
    individuals = kv_store.get("individuals")
    # Randomly sample from individuals until we have size elements
    print(f"Current individuals size: {len(individuals)}")
    prompt = sample_with_replacement(individuals, size, epoch)
    prompt = [item["prompt"] for item in prompt]
    # Print the code with the lowest gap in current individuals and its gap
    # min_gap = min([item["avg_gap"] for item in individuals])
    # min_gap_item = [item for item in individuals if item["avg_gap"] == min_gap][0]
    # print(f"Current min gap: {min_gap}, prompt: {min_gap_item['prompt']}")
    return prompt

@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        timeout_seconds=360,
        test_all=False,
        phase="train"
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.timeout_seconds = timeout_seconds
        self.rgen = random.Random(42)
        self.test_all = test_all
        self.phase = phase


        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
    
    def _compute_score_parallel_with_ray(self, data_sources, solution_strs, ground_truths, extra_infos):
        scores: List[float] = [0.0] * len(solution_strs)
        extra_info_dict: Dict[str, List[float]] = {}  # Key -> list of values for the batch
        print(f"Scoring process started over {len(solution_strs)} samples, waiting for results...")

        futures = []
        uuids = []
        eval_seed = self.rgen.randint(0, 1000000)
        print(f"eval_seed: {eval_seed}")
        for i in range(len(solution_strs)):
            ground_truth = ground_truths[i]
            solution_str = solution_strs[i]
            data_source = data_sources[i]
            extra_info = extra_infos[i]

            pyvrp_id = str(uuid.uuid4())
            uuids.append(pyvrp_id)

            # Submit task to Ray
            future = reward_func_timeout_ray.remote(
                self.compute_score, 
                self.timeout_seconds, 
                data_source, 
                solution_str, 
                ground_truth, 
                pyvrp_id=pyvrp_id, 
                extra_info=extra_info, 
                parent_gap=None, 
                seed=eval_seed, 
                test_all=self.test_all, 
                parent_code=get_code_by_prompt(extra_info['question']),
                phase=self.phase
            )
            futures.append(future)

        default_fail_score = {"score": -1.0, "extra_info": {"is_filter": 1}}  # Default on error which should be filtered
        # Get task results and handle timeout logic
        for i, future in enumerate(futures):
            try:
                # Set timeout for result return. Unlike ProcessPoolExecutor, Ray controls this through the timeout parameter of ray.get
                task_result = ray.get(future, timeout=self.timeout_seconds)

                # Standardize task_result format
                if isinstance(task_result, dict):
                    assert 'extra_info' in task_result, f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                    score_result = task_result
                    # If calculation result is not filtered, ensure correct marking
                    if "is_filter" not in task_result["extra_info"]:
                        score_result["extra_info"].update({"is_filter": 0})
                elif isinstance(task_result, (int, float)):  # Handle scalar return results
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

            # Store final score
            scores[i] = float(score_result.get('score', 0.0))  # Ensure score is float type

            # If extra_info exists, collect it
            if 'extra_info' in score_result and isinstance(score_result['extra_info'], dict):
                for key, value in score_result['extra_info'].items():
                    if key not in extra_info_dict:
                        # Initialize list (e.g., default value 0.0) to match all items
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value
        if wandb.run is not None and self.test_all == False:
            wandb.log({
                "score/hist": wandb.Histogram(scores),
                "score/filter_rate": extra_info_dict["is_filter"].count(1) / len(scores),
                "score/avg_score": np.mean(scores),
                "score/avg_score_unfiltered": np.mean([scores[i] for i in range(len(scores)) if extra_info_dict["is_filter"][i] == 0])
            })
            wandb.run._step -= 1 # reset the step counter sicne `wandb.log()` will increement the step counter
        # Print the solution_str and score of the first data
        print(f"First sample:\n solution_str='{solution_strs[0]}'\n score={scores[0]}")
        global delete_queue
        delete_queue.extend(uuids)
        # Delete all benchmark/pyvrp_{uuid}
        for pyvrp_id in delete_queue:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            path = Path(curr_dir).parent.parent.parent.parent
            target_dir = str(path / "benchmark" / f"pyvrp_{pyvrp_id}")
            os.system(f"rm -rf {target_dir} > /dev/null 2>&1")

            if not os.path.exists(target_dir):
                if pyvrp_id in delete_queue:
                    delete_queue.remove(pyvrp_id)

        return scores, extra_info_dict

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        data_sources = []
        solution_strs = []
        ground_truths = []
        extra_infos = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            data_sources.append(data_source)


            response_ids = data_item.batch["responses"]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            solution_strs.append(response_str)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            ground_truths.append(ground_truth)
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            extra_infos.append(extra_info)



        results = self._compute_score_parallel_with_ray(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )
        scores, extra_infos = results
        for i in range(len(scores)):
            # data_item = data[i]  # DataProtoItem

            # prompt_ids = data_item.batch["prompts"]

            # prompt_length = prompt_ids.shape[-1]

            # valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # response_ids = data_item.batch["responses"]
            # valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            # valid_response_ids = response_ids[:valid_response_length]

            # # decode
            # prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # print(f"prompt_str: {prompt_str}")
            # print(f"response_str: {response_str}")
            # eos_token = self.tokenizer.eos_token
            # if response_str.endswith(eos_token):
            #     response_str = response_str[: -len(eos_token)]

            # ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # result = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )
            # extra_infos: key: [value1, value2...]
            result = {
                "score": scores[i],
            }
            for key in extra_infos:
                result[key] = extra_infos[key][i]
            print(f"*result: {result}")
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            prompt_str = self.tokenizer.decode(prompt_ids[-prompt_length:], skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
