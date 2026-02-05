from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm import GPTConfig, MultiHeadAttention, SmallLanguageModel


@dataclass
class MOEConfig(GPTConfig):
	n_experts: int = 4  # 专家数量
	top_k: int = 2  # 选择前 k 个专家
	shared_expert_num: int = 2  # 共享专家数量


class BasicExpert(nn.Module):
	def __init__(self, config: MOEConfig):
		super().__init__()
		self.hidden_dim = config.hidden_dim

		self.gate = nn.Linear(config.hidden_dim, config.hidden_dim)
		self.up = nn.Linear(self.hidden_dim, self.hidden_dim * 8 // 3)  # magic number 8/3
		self.down = nn.Linear(self.hidden_dim * 8 // 3, self.hidden_dim)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		out = self.dropout(self.down(self.gate(F.silu(self.up(x)))))
		return out


class BasicMOE(nn.Module):
	def __init__(self, config: MOEConfig):
		super().__init__()
		self.config = config
		self.experts = nn.ModuleList([BasicExpert(config) for _ in range(config.n_experts)])
		self.gate = nn.Linear(config.hidden_dim, config.n_experts)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x shape: (batch_size, feature_in)
		expert_weight = self.gate(x)  # shape: (batch_size, n_experts)
		expert_weight = torch.softmax(expert_weight, dim=-1)  # shape: (batch_size, n_experts)

		expert_output = [expert(x) for expert in self.experts]  # 每个 expert 输出 shape: (batch_size, feature_out)
		expert_output = [
			expert_out.unsqueeze(1) for expert_out in expert_output
		]  # 每个 expert_out shape: (batch_size, 1, feature_out)
		expert_output = torch.concat(expert_output, dim=1)  # shape: (batch_size, n_experts, feature_out)

		# 目标 shape : (batch_size, feature_out)
		expert_weight = expert_weight.unsqueeze(1)  # shape: (batch_size, 1, n_experts)
		output = expert_weight @ expert_output  # shape: (batch_size, 1, feature_out)
		return output.squeeze(1)  # shape: (batch_size, feature_out)


class SparseMOE(nn.Module):
	class MOERouter(nn.Module):
		def __init__(self, config: MOEConfig):
			super().__init__()
			self.gate = nn.Linear(config.hidden_dim, config.n_experts)
			self.top_k = config.top_k
			self.n_experts = config.n_experts

		def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
			expert_weight: torch.Tensor = self.gate(x)  # shape: (batch_size * seq_len, n_experts)

			# 计算每个专家概率
			router_prob = F.softmax(expert_weight, dim=-1)  # shape: (batch_size * seq_len, n_experts)

			# 计算 topK 专家输出
			# NOTE topk 是可以反向传播的
			router_weight, selected_idx = torch.topk(
				router_prob, self.top_k, dim=-1
			)  # shape: (batch_size * seq_len, top_k), (batch_size * sql_len, top_k)

			router_weight = router_weight / router_weight.sum(dim=-1, keepdim=True)  # 归一化权重，让其中和重新等于 1
			router_weight = router_weight.to(x.device)

			expert_mask = F.one_hot(
				selected_idx, num_classes=self.n_experts
			).float()  # shape: (batch_size * seq_len, top_k, n_experts)
			expert_mask = expert_mask.permute(2, 1, 0)  # shape: (n_experts, top_k, batch_size * seq_len)

			return expert_weight, router_weight, selected_idx, expert_mask

	def __init__(self, config: MOEConfig):
		super().__init__()
		self.top_k = config.top_k
		self.shared_expert_num = config.shared_expert_num
		self.hidden_dim = config.hidden_dim
		self.n_experts = config.n_experts

		self.experts = nn.ModuleList([BasicExpert(config) for _ in range(config.n_experts)])
		self.gate = SparseMOE.MOERouter(config)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		batch_size, seq_len, hidden_dim = x.size()

		# x reshape 为 (batch_size * seq_len, hidden_dim)
		hidden_state = x.view(batch_size * seq_len, hidden_dim)

		# 做相关专家计算
		expert_weight, router_weight, selected_idx, expert_mask = self.gate(x)

		# 最终目标 shape (batch_size * seq_len, hidden_dim)
		final_hidden_state = torch.zeros(
			[batch_size * seq_len, hidden_dim], device=hidden_state.device, dtype=hidden_state.dtype
		)

		# 遍历每一个专家，把选中的每个专家的 hidden_state 加到 final 中
		for expert_idx in range(self.n_experts):
			expert_layer = self.experts[expert_idx]

			current_expert_mask = expert_mask[expert_idx]  # shape (topk, batch_size * seq_len)

			# idx 表示当前选中的第一个维度的哪个，对应在这里就是 topk 中的哪个
			# top_x 是 token 在 batch_size * seq_len 中的位置索引，对应此处如果 batch_size=2，seq_len=4，那么 top_x 取值 [0-7]
			# idx 用来选中 weight
			# top_x 用来选中 hidden_state
			idx, top_x = torch.where(current_expert_mask)

			current_state = (
				hidden_state.unsqueeze(0)[  # shape (1, batch_size*seq_len, hidden_dim)
					:, top_x:, :
				].reshape(  # shape (1, selected_toekn_number, hidden_dim)
					-1, hidden_dim
				)  # shape (selected_toekn_number, hidden_dim)
			)
			current_state = expert_layer(current_state)

			current_token_router_weight = router_weight[top_x, idx]  # shape (selected_toekn_number)
			current_token_router_weight = current_token_router_weight.unsqueeze(-1)  # shape (selected_toekn_number, 1)

			# 逐个元素相乘
			current_hidden_states = (
				current_state * current_token_router_weight
			)  # 这里 hidden_dim * 1 存在广播，结果 shape(selected_toekn_number, hidden_dim)

			# 对应位置 weight 相加
			final_hidden_state.index_add(0, top_x, current_hidden_states.to(hidden_state.dtype))

		# 还原 final 的 shape
		final_hidden_state = final_hidden_state.reshape(batch_size, seq_len, hidden_dim)

		return final_hidden_state, expert_weight


class SharedExpertMOE(nn.Module):
	def __init__(self, config: MOEConfig) -> None:
		super().__init__()
		self.shared_expert_num = config.shared_expert_num
		self.routed_expert_moe = SparseMOE(config)
		self.shared_experts = nn.ModuleList([BasicExpert(config)] * self.shared_expert_num)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		batch_size, seq_len, hidden_dim = x.size()

		shared_expert_outputs = [expert(x) for expert in self.shared_experts]
		shared_expert_output = torch.stack(
			shared_expert_outputs, dim=0
		)  # shape (shared_expert_num, batch_size, seq_len, hidden_dim)

		shared_expert_output = shared_expert_output.sum(dim=0)  # shape (batch_size, seq_len, hidden_dim)
		sparse_moe_out, router_logits = self.routed_expert_moe(x)  # 选中的专家输出，和选中了哪几个专家
		output = shared_expert_output + sparse_moe_out
		return output, router_logits


class MOEBlock(nn.Module):
	def __init__(self, config: MOEConfig):
		super().__init__()
		self.top_k = config.top_k

		self.ln1 = nn.LayerNorm(config.hidden_dim)
		self.attn = MultiHeadAttention(config)
		self.ln2 = nn.LayerNorm(config.hidden_dim)

		# 使用MOE替换原来的FeedForward
		if config.shared_expert_num > 0:
			self.moe = SharedExpertMOE(config)
		else:
			self.moe = SparseMOE(config)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		x = x + self.attn(self.ln1(x))

		# MOE层的输出和router logits
		moe_out, router_logits = self.moe(self.ln2(x))
		x = x + moe_out

		return x, router_logits


class SmallMOELanguageModel(SmallLanguageModel):
	def __init__(self, config: MOEConfig) -> None:
		super().__init__(config)
		self.blocks = nn.Sequential(*[MOEBlock(config) for _ in range(config.n_layer)])
		self.top_k = config.top_k
		self.n_experts = config.n_experts

	def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
		batch_size, seq_len = idx.size()
		token_embedding = self.token_embedding(idx)  # shape (batch_size, seq_len, n_embd)
		position_embedding = self.position_embedding(
			torch.arange(seq_len, device=idx.device)
		)  # shape (seq_len, n_embd)
		# Position embedding的实际形状是 (seq_len, n_embd)，而不是 (batch_size, seq_len, n_embd)。
		#  当我们将token embedding (batch_size, seq_len, n_embd) 与position embedding (seq_len, n_embd) 相加时，PyTorch会使用广播机制（broadcasting）：
		#   - Position embedding (seq_len, n_embd) 会被自动扩展为 (batch_size, seq_len, n_embd)
		#   - 每个批次中的相同位置都会加上相同的位置嵌入向量
		x = token_embedding + position_embedding

		x, router_logtis = self.blocks(x)
		x = self.ln_final(x)
		logits: torch.Tensor = self.lm_head(x)

		if targets is not None:
			# 计算损失
			batch_size, seq_size, vocab_size = logits.size()
			logits = logits.view(batch_size + seq_size, vocab_size)
			target = targets.view(batch_size + seq_size)
			loss = F.cross_entropy(logits, target)

			mse_loss = F.mse_loss(logits, target)
			aux_loss = self._switch_load_balancing_loss(router_logtis, self.n_experts)

			loss = mse_loss + 0.01 * aux_loss
		else:
			loss = None
		return logits, loss

	def _switch_load_balancing_loss(self, router_logits: torch.Tensor, n_experts: int):
		# 计算路由概率
		router_prob = torch.softmax(router_logits, dim=-1)

		# 获取 topk 专家
		_, selected = torch.topk(router_prob, self.top_k, dim=-1)

		mask = F.one_hot(selected, num_classes=n_experts)

		# 计算每个专家的期望负载 (理想情况下应该是 1/num_experts)
		# expected_load = torch.ones_like(router_prob) / n_experts

		# 计算实际负载 (每个专家处理的token数量除以总token数量)
		# 在batch维度上计算平均值
		actual_load = mask.mean(dim=0)  # [num_experts]

		# 计算auxiliary loss
		# 这会惩罚负载分布与期望负载的差异
		aux_loss = torch.sum(actual_load * router_prob.mean(dim=0)) * n_experts

		# 计算z_loss (可选)
		# 这会惩罚过大的路由logits
		z_loss = torch.mean(torch.square(router_logits))
		z_loss_weight = 0.001  # 可调整的超参数

		# 总损失
		total_loss = aux_loss + z_loss * z_loss_weight

		return total_loss
