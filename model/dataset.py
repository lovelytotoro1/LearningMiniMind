import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())  # 读取所有的文本行，并去除首尾空格
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"  # 在每一行之前添加开始与结束符号
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )  # 对文本向量进行编码, 值得注意的是如果超过最大长度应该怎么处理，截断吗？
        """
            当你使用 tokenizer 对文本进行处理时，如果输入文本的长度超出了 `max_length`，通常会发生以下几种情况，取决于你设置的参数：
            1. **截断（Truncation）**：  
            如果你启用了 `truncation=True`，tokenizer 会自动截断输入文本，使其长度不超过 `max_length`。截断的方式通常是从文本的开头或结尾去掉部分 token（具体行为取决于 tokenizer 配置）。  
            例如，`max_length=512` 时，如果输入文本的 token 数超过 512，超出的部分会被去掉。

            2. **填充（Padding）**：  
            如果启用了 `padding=True`，tokenizer 会对较短的输入进行填充，确保所有文本的长度一致。如果你设置了 `max_length`，那么 tokenizer 会将短于 `max_length` 的文本填充到指定长度。

            3. **错误处理**：  
            如果没有启用截断，并且文本超出了 `max_length`，tokenizer 会报错或者返回原始文本，具体行为取决于框架和库的实现。

            例如，使用 `transformers` 库时，可以像这样进行设置：
            ```python
            encoded = tokenizer(text, padding=True, truncation=True, max_length=512)
            ```

            这样，无论输入文本多长，它都会被自动截断到 512 长度，并在需要时填充至该长度。

            当 padding=True 时，tokenizer 会根据设置的 max_length 或默认的最长序列长度来填充文本，确保所有的输入序列在批处理时具有相同的长度。
            填充的方式通常是将特殊的填充标记（[PAD]）添加到序列的末尾，或者在某些情况下，填充到序列的开头。
        """

        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)  # 预测这段话最后一个词
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids  # 开始符号
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids  # 结束符号

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
            构建符合ChatML格式的对话
            ChatML（Chat Markup Language）是一种用于结构化聊天内容的标记语言，通常用于定义聊天机器人与用户之间的对话格式。
            它可以帮助开发者更清晰地组织对话内容，确保对话的逻辑性和可读性。

            ChatML 的基本结构：
                <conversation>: 表示整个对话的容器。
                <message>: 表示一条消息，通常包含 role 属性来区分消息的发送者（如 user 或 assistant）。
                <text>: 包含消息的具体文本内容。
                <metadata>: 可以包含对话的元数据，如时间戳、用户ID等。
                <action>: 可以定义对话中的操作或指令，如调用API、跳转等。

            除了ChatML，还有其他一些用于结构化聊天内容的标记语言或格式。以下是一些常见的结构化聊天内容标记语言或格式：

            1. Chatito： Chatito 是一种用于生成训练数据的标记语言，主要用于构建自然语言理解（NLU）模型的训练数据集。它支持定义意图、实体和对话流。
                %[greet]
                    ~[hi]
                    ~[hello]
                    ~[hey]
                ~[hi]
                    hi
                    hello there
                    hey
                ~[hello]
                    hello
                    hi
                    hey there
            2. Rasa NLU Training Data Format： Rasa NLU 是一个用于构建自然语言理解（NLU）模型的开源工具。它使用一种特定的训练数据格式来定义意图、实体和对话流。
                version: "3.0"
                nlu:
                - intent: greet
                    examples: |
                    - 你好
                    - 早上好
                    - 嗨
                - intent: goodbye
                    examples: |
                    - 再见
                    - 拜拜
                    - 下次见
            3. Dialogflow Agent： Dialogflow 是一个用于构建聊天机器人的平台。它使用一种特定的代理格式来定义意图、实体和对话流。
            4. OpenAI Chat Completion Format (JSON)：OpenAI 的 GPT 模型使用 JSON 格式来定义对话内容，通常用于 API 调用。每条消息包含 role（角色）和 content（内容）
                {
                    "messages": [
                        {"role": "system", "content": "你是一个有帮助的助手。"},
                        {"role": "user", "content": "今天的天气怎么样？"},
                        {"role": "assistant", "content": "今天的天气晴朗，气温在20到25摄氏度之间。"}
                    ]
                }
            
            5. Markdown with Metadata：一些聊天机器人框架支持使用 Markdown 格式，并结合元数据（如 YAML 或 JSON）来定义对话内容。
                ---
                role: user
                timestamp: 2023-10-05T14:30:00Z
                ---
                你好，我想订一张机票。

            ChatML：适合自定义对话结构。
            Rasa/Dialogflow：适合 NLU 和对话管理。
            AIML：适合基于规则的聊天机器人。
            JSON/YAML：适合 API 驱动的对话系统。
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

        """
            一份数据分俩条
            chosen：用户选择的对话
                ['role': 'user, 'content': '']
                ['role': 'system', 'content': '']
            rejected：用户拒绝的对话
                ['role': 'user, 'content': '']
                ['role': 'system', 'content': '']

        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


if __name__ == "__main__":
    pass
