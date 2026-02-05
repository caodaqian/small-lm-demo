import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
	block_size: int = 512  # 单个序列的最大长度
	batch_size: int = 4  # 每次训练的序列数量
	n_layer: int = 2  # Transformer块的数量
	n_head: int = 12  # 注意力头的数量
	n_embd: int = 768  # 嵌入向量的维度
	hidden_dim: int = n_embd
	head_size: int = n_embd // n_head  # 每个注意力头的维度
	dropout: float = 0.01
	vocab_size: int = 50257  # 词表大小


class SingleHeadAttention(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()

		self.head_size: int = config.head_size

		self.query: nn.Linear = nn.Linear(config.hidden_dim, config.head_size, bias=True)
		self.key: nn.Linear = nn.Linear(config.hidden_dim, config.head_size, bias=True)
		self.value: nn.Linear = nn.Linear(config.hidden_dim, config.head_size, bias=True)

		# 通过 register buffer 注册一个缓冲区，用于存储注意力掩码
		self.register_buffer("attention_mask", torch.tril(torch.ones(config.block_size, config.block_size)))

		self.dropout: nn.Dropout = nn.Dropout(config.dropout)

	def forward(self, x: torch.Tensor) -> None:
		batch_size, seq_size, hidden_dim = x.size()

		k: torch.Tensor = self.key(x)  # shape: (batch_size, seq_size, head_size)
		q: torch.Tensor = self.query(x)  # shape: (batch_size, seq_size, head_size)
		v = self.value(x)  # shape: (batch_size, seq_size, head_size)

		# 计算注意力权重 Q @ K^T
		attention_weights = q @ k.transpose(-2, -1)  # shape: (batch_size, seq_size, seq_size)
		# 应用注意力掩码，防止模型看到未来的信息
		attention_weights = attention_weights.masked_fill(self.attention_mask[:seq_size, :seq_size] == 0, float("-inf"))  # pyright: ignore[reportIndexIssue]
		# softmax 归一化，并缩放
		attention_weights = F.softmax(attention_weights, dim=-1) / math.sqrt(self.head_size)
		# 应用 dropout
		attention_weights = self.dropout(attention_weights)

		output = attention_weights @ v  # shape: (batch_size, seq_size, head_size)
		return output


class MultiHeadAttention(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()

		self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
		self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 并行计算所有注意力头的输出
		heads_output = torch.cat([head(x) for head in self.heads], dim=-1)  # shape: (batch_size, seq_size, hidden_dim)
		# 线性变换和 dropout
		output = self.dropout(self.proj(heads_output))
		return output


class MatrixRotationMultiHeadAttention(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()

		self.hidden_dim = config.hidden_dim
		self.head_size = config.head_size
		self.head_dim = self.hidden_dim // self.head_size

		self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, seq_size, hidden_dim = x.size()

		# (batch_size, seq_size, hidden_dim) -> (batch_size, seq_size, head_size, head_dim) -> (batch_size, head_size, seq_size, head_dim)
		q = self.q_proj(x).view(batch_size, seq_size, self.head_size, self.head_dim).transpose(1, 2)
		k = self.k_proj(x).view(batch_size, seq_size, self.head_size, self.head_dim).transpose(1, 2)
		v = self.v_proj(x).view(batch_size, seq_size, self.head_size, self.head_dim).transpose(1, 2)

		attention_weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
		attention_weight = attention_weight.masked_fill(
			torch.tril(torch.ones(seq_size, seq_size).to(x.device).view(batch_size, self.head_size, seq_size, seq_size))
			== 0,
			float("-inf"),
		)
		attention_weight = F.softmax(attention_weight, dim=-1)
		attention_weight = self.dropout(attention_weight)

		output = attention_weight @ v  # shape: (batch_size, head_size, seq_size, head_dim)
		output = (
			output.transpose(1, 2).contiguous().view(batch_size, seq_size, hidden_dim)
		)  # shape: (batch_size, seq_size, hidden_dim)
		output = self.out_proj(output)
		return output


class FeedForward(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()

		self.net = nn.Sequential(
			nn.Linear(config.hidden_dim, 4 * config.hidden_dim),  # 如果用 SwiGLU， 只需要升维到 8/3 hidden_dim
			nn.GELU(),
			nn.Linear(4 * config.hidden_dim, config.hidden_dim),
			nn.Dropout(config.dropout),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class Block(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()

		self.ln1 = nn.LayerNorm(config.hidden_dim)
		self.attn = MultiHeadAttention(config)
		self.ln2 = nn.LayerNorm(config.hidden_dim)
		self.ffwd = FeedForward(config)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x


class SmallLanguageModel(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()

		# 当前 LLM 升级点：
		# 1. postion embedding 改为 RoPE (Rotary Positional Embeddings)
		# 2. norm -> rmsnorm
		# 3. mlp -> swiglu
		# 4. mha -> gqa

		self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
		self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
		self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
		self.ln_final = nn.LayerNorm(config.n_embd)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.block_size = config.block_size

		# 现在模型会用 tie weight 减少参数
		# 将 lm_head 的权重与 token_embedding 的权重绑定
		# 之前所有的 Linear 层 (4 -> 8)，weight 是 (8 * 4)
		self.token_embedding.weight = self.lm_head.weight

	def _init_weights(self, module: nn.Module) -> None:
		# 初始化模型权
		if isinstance(module, nn.Linear):
			## 初始化为高斯分布
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(
		self, idx: torch.Tensor, targets: torch.Tensor | None = None
	) -> Tuple[torch.Tensor, torch.Tensor | None]:
		"""前向传播函数
		Args:
			idx (torch.Tensor): 输入的 token 索引，shape: (batch_size, seq_size)
			targets (torch.Tensor, optional): 目标 token 索引，shape: (batch_size, seq_size). Defaults to None.
		Returns:
			torch.Tensor: 如果提供了 targets，返回损失值；否则返回 logits。
		"""
		batch_size, seq_size = idx.size()  # shape: (batch_size, seq_size)
		token_embedding = self.token_embedding(idx)  # shape: (batch_size, seq_size, n_embd)
		position_embedding = self.position_embedding(
			torch.arange(seq_size, device=idx.device)
		)  # shape: (seq_size, n_embd)
		# NOTE Question: token_emb 和 position_embedding 为什么能直接相加？
		x = token_embedding + position_embedding  # shape: (batch_size, seq_size, n_embd)

		x = self.blocks(x)  # shape: (batch_size, seq_size, n_embd)
		x = self.ln_final(x)  # shape: (batch_size, seq_size, n_embd)
		logits: torch.Tensor = self.lm_head(x)  # shape: (batch_size, seq_size, vocab_size)

		if targets is not None:
			# 计算交叉熵损失
			batch_size, seq_size, vocab_size = logits.size()
			logits = logits.view(batch_size * seq_size, vocab_size)
			target = targets.view(batch_size * seq_size)
			loss = F.cross_entropy(logits, target)
		else:
			loss = None
		return logits, loss

	def generate(
		self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 20
	) -> torch.Tensor:
		# idx shape: (batch_size, seq_size)
		for _ in range(max_new_tokens):
			# 截断输入到 block_size
			idx_cond = idx[:, -self.block_size :]
			# 前向传播
			logits, _ = self.forward(idx_cond)
			# 取最后一个时间步的 logits
			logits = logits[:, -1, :] / temperature  # shape: (batch_size, vocab_size)

			# top-k 采样
			vocab_size = logits.size(-1)
			if top_k > 0:
				top_k = min(top_k, vocab_size)  # 防止 top_k 大于 vocab_size
				# 获取 top_k 的 logits 和索引
				top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
				# 创建一个新的 logits 张量，其他位置设为 -inf
				new_logits = torch.full_like(logits, float("-inf"))
				new_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
				logits = new_logits

			# temperature 采样
			if temperature > 0:
				logits = logits / temperature  # 对 logits 进行缩放

			# 计算概率分布
			probs = F.softmax(logits, dim=-1)  # shape: (batch_size, vocab_size)
			# 从分布中随机采样下一个 token
			next_token = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)
			# 将下一个 token 拼接到序列中
			idx = torch.cat((idx, next_token), dim=1)  # shape: (batch_size, seq_size + 1)
		return idx
