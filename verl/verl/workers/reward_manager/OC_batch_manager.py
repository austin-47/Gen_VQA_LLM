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
import ray

from verl import DataProto
from bert_score import BERTScorer
from verl.utils.reward_score.oc_reward_batch import batch_compute_score

class OCBatchRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key='data_source', **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or batch_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        
        _bert_path="../llms/bert_series/bert-base-uncased"
        _baseline_path="../llms/bert_series/bert-base-uncased.tsv"
        self.bert_model = BERTScorer(
            model_type=_bert_path,
            lang="en",
            rescale_with_baseline=True,
            num_layers=9,
            # device="cuda",
            baseline_path=_baseline_path,
        )
    
    def verify(self, data):
        prompts = data.batch['prompts']
        responses = data.batch['responses']
        mask = data.batch['attention_mask']
        prompt_len = prompts.size(-1)
        valid_lens = mask[:, prompt_len:].sum(dim=-1)

        decoded = []
        for i, L in enumerate(valid_lens.tolist()):
            decoded.append(self.tokenizer.decode(responses[i, :L], skip_special_tokens=True))
        gts = [item.non_tensor_batch['reward_model'].get('ground_truth') for item in data]
        sources = data.non_tensor_batch.get(self.reward_fn_key, [None] * len(data))
        extras = data.non_tensor_batch.get('extra_info', [None] * len(data))

        return self.compute_score(
            self.bert_model, sources, decoded, gts, extras,
            **{k: v for k, v in self.reward_kwargs.items() if k != 'bert_model'}
        )

    def __call__(self, data, return_dict=False):
        # If precomputed rm_scores
        if 'rm_scores' in data.batch:
            if return_dict:
                return {'reward_tensor': data.batch['rm_scores']}
            return data.batch['rm_scores']

        prompts = data.batch['prompts']
        print(prompts.device)
        responses = data.batch['responses']
        mask = data.batch['attention_mask']
        prompt_len = prompts.size(-1)
        valid_lens = mask[:, prompt_len:].sum(dim=-1)

        # Compute scores
        scores = self.verify(data)

        # Build reward tensor
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        rewards = []
        for i, sc in enumerate(scores):
            L = valid_lens[i].item()
            rewards.append(sc)
            if L > 0:
                reward_tensor[i, L-1] = sc

        # Assign acc
        data.batch['acc'] = torch.tensor(rewards, dtype=torch.float32, device=prompts.device)

        # Print examples up to num_examine per data source
        already_printed = {}
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, [None] * len(data))
        for i in range(len(data)):
            src = data_sources[i]
            count = already_printed.get(src, 0)
            if count < self.num_examine:
                L = valid_lens[i].item()
                prompt_str = self.tokenizer.decode(prompts[i], skip_special_tokens=True)
                response_str = self.tokenizer.decode(responses[i, :L], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch['reward_model'].get('ground_truth')
                print(f"[prompt] {prompt_str}")
                print(f"[response] {response_str}")
                print(f"[ground_truth] {ground_truth}")
                print(f"[score] {scores[i]}")
                already_printed[src] = count + 1

        if return_dict:
            return {'reward_tensor': reward_tensor, 'reward_extra_info': {}}
        return reward_tensor