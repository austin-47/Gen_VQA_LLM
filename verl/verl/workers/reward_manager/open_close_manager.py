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
from bert_score import BERTScorer
from verl.utils.reward_score.open_close_reward import my_compute_score


class OpenCloseRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console [打印的批次数量]
        self.compute_score = compute_score or my_compute_score
        self.reward_fn_key = reward_fn_key
        _bert_path = "../llms/bert_series/deberta-large-mnli"
        _baseline_path="./llms/bert_series/baselines_tsv/deberta-large-mnli.tsv"
        self.bert_model = BERTScorer(
            model_type=_bert_path,
            lang="en",
            rescale_with_baseline=True,
            num_layers=18,
            baseline_path=_baseline_path,
        )
        
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        ## StringKeys(dict_keys(['prompts', 'position_ids', 'responses', 'attention_mask', 'input_ids']))
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32) ## [610,2048]
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)): ## 601
            data_item = data[i]  # DataProtoItem
            #data_item: non_tensor_batch={'data_source': 'hiyouga/geometry3k', 'ability': 'math', 'reward_model': {'ground_truth': '48', 'style': 'rule'}, 'extra_info': {'answer': '48', 'index': 0, 'question': '<image>Chords $\\overline{A C}$ and $\\overline{D F}$ are equidistant from the center. If the radius of $\\odot G$ is 26 find $A C$', 'split': 'test'}, 'index': 0, 'multi_modal_inputs': {'pixel_values'
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] ## 207

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ## '48'
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            ## 'hiyouga/geometry3k'
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                bert_model=self.bert_model,
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            ## advantage is broadcast to the token level
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
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
