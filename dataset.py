import json

import tiktoken
from torch.utils.data import Dataset


class MyDataset(Dataset):
	def __init__(self, data_path: str, block_size: int):
		self.enc = tiktoken.get_encoding("gpt2")
		self.block_size = block_size

		# 添加特殊文本结束标记
		self.eos_token = self.enc.eot_token  # GPT "<|endoftext|>"

		encoded_lines = []
		with open(data_path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				text = json.loads(line)["text"]  # 默认每一行是一个 JSON 对象，提取 "text" 字段
				# 编码文本并添加结束标记
				encoded_line = self.enc.encode(text) + [self.eos_token]
				encoded_lines.append(encoded_line)

		# 进行 chunk 划分
		self.encoded_data = []
		for i in range(0, len(encoded_lines), self.block_size):
			chunk = encoded_lines[i : i + self.block_size + 1]  # 取 block_size + 1 长度，方便 shift label
			if len(chunk) < self.block_size + 1:
				chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))  # 填充 eos_token padding
			self.encoded_data.append(chunk)

	def __len__(self):
		return len(self.encoded_data)

	def __getitem__(self, idx):
		chunk = self.encoded_data[idx]
		input_ids = chunk[:-1]  # 输入部分
		target_ids = chunk[1:]  # 目标部分，向右移动一位
		return input_ids, target_ids
