import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)  # 怎么只有一个epoch
    """
    语言模型（如GPT系列）的全监督微调（fine-tuning）所需的epoch数量通常取决于多个因素，包括数据集的大小、模型的规模、学习率、批次大小等。
    数据集大小：如果你的数据集相对较小，可能只需要几轮训练（比如5-10个epoch）就能得到较好的效果；如果数据集很大，可能需要更多的epoch来充分学习。
    学习率：较高的学习率可能导致模型过早收敛，需要较少的epoch；较低的学习率则可能需要更多的epoch。
    模型规模：较大的预训练模型可能不需要太多的epoch就能从数据中捕捉到有用的信息，但它们通常也需要更多的计算资源。
    过拟合问题：对于较小的训练集，训练太多的epoch可能会导致过拟合，这时可以通过早停（early stopping）来避免。
    通常情况下，很多研究和实践中的经验值是在微调时使用大约 3到5个epoch，特别是在一个中等大小的数据集上。但对于一些大规模数据集，可能需要 10个epoch 或更多，或者采用 逐步训练 的策略。
    """

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../datasets/minimind_dataset/sft_mini_512.jsonl")

    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)  # 与pretrain.py中的init_model函数相同, 语言模型初始化换成了Pretrain的模型

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)  # 返回结构化的聊天数据，ChatML格式
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    """
        _ddp_params_and_buffers_to_ignore 是 PyTorch 中与 Distributed Data Parallel (DDP) 相关的一个内部机制，用于指定在进行模型的分布式训练时，哪些模型参数或缓存（buffers）不参与 DDP 的通信过程。
        在分布式训练中，DDP 会自动同步不同计算设备（例如多张 GPU）上的梯度和模型参数。但是，有时我们可能希望在某些情况下，忽略某些不需要同步的参数或缓冲区。
        这个功能通常用于以下几种情况：
        冻结某些层的参数：例如，如果你只想训练部分网络层，或者在迁移学习中只训练新加入的层，可以使用这个机制来忽略其他层的参数。
        不需要同步的缓冲区：有些参数或缓冲区（如 batch normalization 的 running mean 和 running variance）可能不需要每个设备之间的同步。
        PyTorch 在进行分布式训练时，会默认同步模型的所有可训练参数，但你可以通过 _ddp_params_and_buffers_to_ignore 来指定某些不参与同步的参数或缓存。
        不过，_ddp_params_and_buffers_to_ignore 是一个内部机制，通常用户不需要手动使用它，除非你在进行一些非常具体的定制操作。
    """
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
