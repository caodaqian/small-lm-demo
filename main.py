import tiktoken
import torch
from torch.utils.data import DataLoader

from dataset import MyDataset
from slm import GPTConfig, SmallLanguageModel


def train(model, optimizer, scheduler, train_loader, device):
	model.train()
	total_loss = 0.0

	for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
		# 将数据移动到设备上
		input_ids = input_ids.to(device)
		target_ids = target_ids.to(device)

		# 前向传播
		logits, loss = model(input_ids, targets=target_ids)

		# 反向传播和优化
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()  # 更新学习率

		total_loss += loss.item()

		if batch_idx % 100 == 0:
			print(f"Epoch [{epoch + 1}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
		return total_loss / len(train_loader)


def eval(model, eval_loader, device):
	model.eval()  # 去除 dropout 等
	total_loss = 0.0

	with torch.no_grad():
		for input_ids, target_ids in eval_loader:
			# 将数据移动到设备上
			input_ids = input_ids.to(device)
			target_ids = target_ids.to(device)

			# 前向传播
			_, loss = model(input_ids, targets=target_ids)
			total_loss += loss.item()
	return total_loss / len(eval_loader)


if __name__ == "__main__":
	config = GPTConfig()

	model = SmallLanguageModel(config)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = model.to(device)
	print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6} M")

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

	# 加载数据集
	train_dataset = MyDataset(data_path="data/train.jsonl", block_size=config.block_size)
	train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

	enc = tiktoken.get_encoding("gpt2")
	for epoch in range(10):
		train_loss = train(model, optimizer, scheduler, train_loader, device)
		eval_loss = eval(model, eval_loader, device)
		print(f"Epoch [{epoch + 1}/10], Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

		# 保存模型检查点
		checkpoint = {
			"epoch": epoch + 1,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"eval_loss": eval_loss,
		}
		torch.save(checkpoint, f"model_epoch_{epoch + 1}.pt")

		# 示例生成
		input = enc.encode("今天天气如何？")
		input = torch.tensor(input, dtype=torch.long).unsqueeze(0)
		output = model.generate(input, 512)
		output = enc.decode(output[0].tolist())
		print(f"Generated text: {output}")
