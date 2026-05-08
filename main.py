from __future__ import print_function

import os
import time
import math
import copy
from typing import Optional, Tuple, Dict, Any

os.environ.setdefault('CUDA_VISIBLE_DEVICES', "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from tools.lib import dump_json, set_seed, count_parameters
from tools.prototypes import PrototypeBank
from SWR_DiMC.spiking_model import SWR_DiMC_general
from SWR_DiMC.Hyperparameters import args


def _env_float(name: str, default: float, aliases=None) -> float:
    if aliases is None:
        aliases = []
    v = os.getenv(name, None)
    if v is None:
        for a in aliases:
            v = os.getenv(a, None)
            if v is not None:
                break
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_int(name: str, default: int, aliases=None) -> int:
    if aliases is None:
        aliases = []
    v = os.getenv(name, None)
    if v is None:
        for a in aliases:
            v = os.getenv(a, None)
            if v is not None:
                break
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False, aliases=None) -> bool:
    if aliases is None:
        aliases = []
    v = os.getenv(name, None)
    if v is None:
        for a in aliases:
            v = os.getenv(a, None)
            if v is not None:
                break
    if v is None:
        return bool(default)
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def _env_str(name: str, default: str, aliases=None) -> str:
    if aliases is None:
        aliases = []
    v = os.getenv(name, None)
    if v is None:
        for a in aliases:
            v = os.getenv(a, None)
            if v is not None:
                break
    if v is None:
        return str(default)
    return str(v)


set_seed(1111)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('GPU is available' if device == 'cuda' else 'GPU is not available')

snn_ckp_dir = './exp/SWR_DiMC/checkpoint/'
snn_rec_dir = './exp/SWR_DiMC/record/'
data_path = './data'

num_epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size



class MNIST(torch.utils.data.Dataset):

    def __init__(self, mnist, perm: torch.Tensor, in_size=1):
        self.mnist = mnist
        self.perm = perm
        self.in_size = int(in_size)
        assert 784 % self.in_size == 0, "in_size must divide 784 exactly."

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        unrolled = img.reshape(-1)
        permuted = unrolled[self.perm]
        permuted = permuted.reshape(-1, self.in_size)
        return permuted, label


def _load_base_permutation() -> torch.Tensor:
    try:
        p0 = torch.load("./ps_data/permutation.pt").long()
        if p0.numel() != 784:
            raise ValueError("permutation.pt must have 784 elements")
        return p0
    except Exception:
        g = torch.Generator().manual_seed(1234)
        return torch.randperm(784, generator=g)


def _make_permutations(num_tasks: int, seed0: int = 1234) -> list:
    perms = []
    perms.append(_load_base_permutation())
    for t in range(1, num_tasks):
        g = torch.Generator().manual_seed(seed0 + t)
        perms.append(torch.randperm(784, generator=g))
    return perms


transform = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
mnist_val = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

T_steps = 784 // int(args.in_size)
net = SWR_DiMC_general(T=T_steps).to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)


ENABLE_PROTOTYPES = _env_bool('ENABLE_PROTOTYPES', True)
NUM_PROTOTYPES = _env_int('NUM_PROTOTYPES', 256)
PROTO_MOMENTUM = _env_float('PROTO_MOMENTUM', 0.95)
PROTO_WARMUP_EPOCHS = _env_int('PROTO_WARMUP_EPOCHS', 1)

LAMBDA_SSL = _env_float('LAMBDA_SSL', 0.05)
SSL_TEMP = _env_float('SSL_TEMP', 0.1)
SSL_MARGIN = _env_float('SSL_MARGIN', 0.05)
SSL_MIN_SIM = _env_float('SSL_MIN_SIM', 0.0)

REPLAY_DISTILL_MODE = os.getenv('REPLAY_DISTILL_MODE', 'ema').lower()  # 'ema' | 'labels' | 'none'
ENABLE_LABEL_DISTILL = (REPLAY_DISTILL_MODE == 'labels')
ENABLE_EMA_TEACHER = _env_bool('ENABLE_EMA_TEACHER', True)
EMA_MOMENTUM = _env_float('EMA_MOMENTUM', 0.999)
REPLAY_DISTILL_TEMP = _env_float('REPLAY_DISTILL_TEMP', 1.0)

net_ema = None
if ENABLE_EMA_TEACHER:
    net_ema = copy.deepcopy(net).to(device)
    net_ema.eval()
    for p in net_ema.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def _ema_update(ema_model: nn.Module, model: nn.Module, m: float) -> None:
    if ema_model is None:
        return
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v in esd.items():
        src = msd[k]
        if torch.is_floating_point(v):
            v.mul_(m).add_(src, alpha=(1.0 - m))
        else:
            v.copy_(src)


proto_bank = None
proto_label_counts = None

if ENABLE_PROTOTYPES:
    emb_dim = int(args.fc[-1])
    proto_bank = PrototypeBank(NUM_PROTOTYPES, emb_dim, device=device, momentum=PROTO_MOMENTUM, init_from_first_batch=True)
    if ENABLE_LABEL_DISTILL:
        proto_label_counts = torch.zeros(NUM_PROTOTYPES, int(args.out_size), device=device)
    else:
        proto_label_counts = None
    print(f"[ProtoBank] enabled: K={NUM_PROTOTYPES}, D={emb_dim}, momentum={PROTO_MOMENTUM}, warmup={PROTO_WARMUP_EPOCHS}")


@torch.no_grad()
def _update_proto_label_hist(idx: torch.Tensor, targets: torch.Tensor):
    if proto_label_counts is None:
        return
    onehot = F.one_hot(targets, num_classes=int(args.out_size)).float()
    proto_label_counts.index_add_(0, idx, onehot)


@torch.no_grad()
def _sample_proto_ids_by_usage(B: int) -> Optional[torch.Tensor]:
    if proto_bank is None:
        return None
    counts = getattr(proto_bank, "counts", None)
    if counts is None:
        return None
    w = counts.clamp_min(0)
    if w.sum() <= 0:
        return None
    p = (w / w.sum()).clamp_min(1e-12)
    return torch.multinomial(p, num_samples=B, replacement=True)


LAMBDA_ENERGY = _env_float('LAMBDA_ENERGY', 0.0, aliases=['MBDA_ENERGY'])
ENERGY_TARGET = _env_float('ENERGY_TARGET', 0.0)

energy_epoch_record = []


ENABLE_REPLAY = _env_bool('ENABLE_REPLAY', True)
REPLAY_WARMUP_EPOCHS = _env_int('REPLAY_WARMUP_EPOCHS', 1)
REPLAY_EVERY_EPOCHS = _env_int('REPLAY_EVERY_EPOCHS', 1)

REPLAY_STEPS = _env_int('REPLAY_STEPS', 200)
REPLAY_BATCH = _env_int('REPLAY_BATCH', 256)

REPLAY_K = _env_int('REPLAY_K', 5)
REPLAY_NOISE_MAX = _env_float('REPLAY_NOISE_MAX', 0.6)
REPLAY_NOISE_MIN = _env_float('REPLAY_NOISE_MIN', 0.05)
REPLAY_CHAIN_MIX = _env_float('REPLAY_CHAIN_MIX', 0.6)
REPLAY_INTERMEDIATE_LOSS = _env_bool('REPLAY_INTERMEDIATE_LOSS', True)

REPLAY_LR = _env_float('REPLAY_LR', 1e-3)
LAMBDA_REPLAY_DENOISE = _env_float('LAMBDA_REPLAY_DENOISE', 1.0)
LAMBDA_REPLAY_DISTILL = _env_float('LAMBDA_REPLAY_DISTILL', 0.5)

REPLAY_UPDATE_FC4 = _env_bool('REPLAY_UPDATE_FC4', True)
REPLAY_UPDATE_EMB_PROJ = _env_bool('REPLAY_UPDATE_EMB_PROJ', True)
LAMBDA_REPLAY_EMB_PROJ_L2 = _env_float('LAMBDA_REPLAY_EMB_PROJ_L2', 1e-4)

REPLAY_TB = _env_int('REPLAY_TB', 20)
REPLAY_BURST_START = _env_int('REPLAY_BURST_START', 8)
REPLAY_BURST_LEN = _env_int('REPLAY_BURST_LEN', 5)
REPLAY_DEN_HMULT = _env_int('REPLAY_DEN_HMULT', 2)
REPLAY_DEN_TDIM = _env_int('REPLAY_DEN_TDIM', 32)

DEN_THRESH = _env_float('REPLAY_DEN_THRESH', float(args.thresh))
DEN_LENS = _env_float('REPLAY_DEN_LENS', float(args.lens))
DEN_DECAY = _env_float('REPLAY_DEN_DECAY', float(args.decay))

REPLAY_SPIKE_BUDGET = _env_float('REPLAY_SPIKE_BUDGET', 0.0)
LAMBDA_REPLAY_SPIKE_BUDGET = _env_float('LAMBDA_REPLAY_SPIKE_BUDGET', 0.1)


REPLAY_TRIGGER_MODE = _env_str('REPLAY_TRIGGER_MODE', 'auto').lower()
REPLAY_TRIGGER_COOLDOWN = _env_int('REPLAY_TRIGGER_COOLDOWN', 1)
REPLAY_TRIGGER_MAX_GAP = _env_int('REPLAY_TRIGGER_MAX_GAP', 3)

REPLAY_TRIGGER_SSL_KEEP_LT = _env_float('REPLAY_TRIGGER_SSL_KEEP_LT', 0.25)
REPLAY_TRIGGER_TOP1_SIM_LT = _env_float('REPLAY_TRIGGER_TOP1_SIM_LT', 0.25)
REPLAY_TRIGGER_MARGIN_LT = _env_float('REPLAY_TRIGGER_MARGIN_LT', 0.05)
REPLAY_TRIGGER_ACTIVE_LT = _env_float('REPLAY_TRIGGER_ACTIVE_LT', 0.20)

REPLAY_TRIGGER_VERBOSE = _env_bool('REPLAY_TRIGGER_VERBOSE', False)



def _make_sigma_schedule(K: int, sigma_max: float, sigma_min: float, device: str):
    if K <= 1:
        return torch.tensor([sigma_max], device=device)
    return torch.linspace(sigma_max, sigma_min, steps=K, device=device)


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(DEN_THRESH).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (input - DEN_THRESH).abs() < DEN_LENS
        return grad_input * temp.float()


spike_fn = _SpikeFn.apply


def _lif_step(x: torch.Tensor, mem: torch.Tensor, spk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mem = mem * DEN_DECAY * (1.0 - spk) + x
    spk = spike_fn(mem)
    return mem, spk


class SpikingBurstDenoiser(nn.Module):

    def __init__(self, dim: int, K: int, hidden_mult: int = 2, tdim: int = 32, Tb: int = 20):
        super().__init__()
        self.dim = int(dim)
        self.K = int(max(K, 1))
        self.Tb = int(max(Tb, 1))
        h = max(self.dim * int(hidden_mult), 64)
        self.h = int(h)
        self.t_embed = nn.Embedding(self.K, int(tdim))
        self.fc_in = nn.Linear(self.dim + int(tdim), self.h)
        self.fc_out = nn.Linear(self.h, self.dim)

        start = int(max(REPLAY_BURST_START, 0))
        length = int(max(REPLAY_BURST_LEN, 0))
        end = min(start + length, self.Tb)
        m = torch.zeros(self.Tb, dtype=torch.float32)
        if end > start:
            m[start:end] = 1.0
        self.register_buffer("burst_mask_t", m)

    def forward(self, z: torch.Tensor, t_idx: torch.Tensor, return_stats: bool = False):
        te = self.t_embed(t_idx)
        x = torch.cat([z, te], dim=1)
        cur = self.fc_in(x)

        B = z.size(0)
        mem = torch.zeros(B, self.h, device=z.device)
        spk = torch.zeros(B, self.h, device=z.device)
        spk_sum = torch.zeros(B, self.h, device=z.device)

        for s in range(self.Tb):
            mem, spk = _lif_step(cur, mem, spk)
            spk = spk * self.burst_mask_t[s]
            spk_sum = spk_sum + spk

        rate = spk_sum / float(self.Tb)
        out = self.fc_out(rate)

        if not return_stats:
            return out

        spikes_per_sample = spk_sum.sum(dim=1)
        spike_keep = (spk_sum > 0).float().mean()
        spike_rate_mean = spk_sum.mean() / float(self.Tb)
        return out, spikes_per_sample, spike_keep, spike_rate_mean


denoiser = None
replay_optimizer = None
sigma_schedule = None
emb_proj_eye = None

if ENABLE_REPLAY and proto_bank is not None:
    denoiser = SpikingBurstDenoiser(int(args.fc[-1]), K=REPLAY_K, hidden_mult=REPLAY_DEN_HMULT, tdim=REPLAY_DEN_TDIM, Tb=REPLAY_TB).to(device)
    replay_params = list(denoiser.parameters())
    if REPLAY_UPDATE_FC4:
        replay_params += list(net.fc4.parameters())
    if REPLAY_UPDATE_EMB_PROJ and hasattr(net, 'emb_proj'):
        replay_params += list(net.emb_proj.parameters())
        with torch.no_grad():
            emb_proj_eye = torch.eye(net.emb_proj.weight.shape[0], device=device)

    replay_optimizer = torch.optim.Adam(replay_params, lr=REPLAY_LR)
    sigma_schedule = _make_sigma_schedule(REPLAY_K, REPLAY_NOISE_MAX, REPLAY_NOISE_MIN, device)
    print(f"[Replay-K] enabled: K={REPLAY_K}, Tb={REPLAY_TB}, burst=[{REPLAY_BURST_START},{REPLAY_BURST_START+REPLAY_BURST_LEN}) "
          f"steps={REPLAY_STEPS}, batch={REPLAY_BATCH}, mix={REPLAY_CHAIN_MIX}, "
          f"lr={REPLAY_LR}, update_fc4={REPLAY_UPDATE_FC4}, update_emb_proj={REPLAY_UPDATE_EMB_PROJ}, "
          f"spike_budget_lambda={LAMBDA_REPLAY_SPIKE_BUDGET}, spike_budget={REPLAY_SPIKE_BUDGET}")


loss_train_record = []
loss_test_record = []
fire_rate_record = []

ssl_loss_record = []
ssl_keep_ratio_record = []
proto_stats_record = []

ssl_top1_sim_record = []
ssl_margin_mean_record = []

replay_triggered_record = []
replay_trigger_reason_record = []
replay_trigger_score_record = []

replay_denoise_record = []
replay_distill_record = []
replay_total_record = []
replay_cos_progress_record = []

replay_relu_keep_record = []
replay_active_units_record = []
replay_forwards_record = []

replay_emb_proj_reg_record = []

replay_spikes_per_sample_record = []
replay_spike_rate_record = []
replay_spike_budget_loss_record = []


def train_one_epoch(trainloader, epoch: int):
    net.train()
    correct = 0
    total = 0
    loss_sum = 0.0

    ssl_loss_sum = 0.0
    ssl_keep_sum = 0.0
    ssl_batches = 0

    ssl_top1_sum = 0.0
    ssl_margin_sum = 0.0
    ssl_stat_batches = 0

    energy_sum = 0.0
    energy_loss_sum = 0.0
    energy_batches = 0

    t0 = time.time()
    for inputs, targets in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs, aux = net(inputs)
        ce_loss = criterion(outputs, targets)

        ssl_loss = torch.tensor(0.0, device=device)
        keep_ratio = 0.0
        emb = None
        idx_upd = None

        if proto_bank is not None and isinstance(aux, dict) and aux.get('emb', None) is not None:
            emb = aux['emb']
            with torch.no_grad():
                proto_bank.maybe_init(emb.detach())

            emb_n = F.normalize(emb, p=2, dim=1, eps=1e-12)
            sims = emb_n @ proto_bank.prototypes.t()
            idx_upd = sims.argmax(dim=1)

            with torch.no_grad():
                top2_all = sims.topk(k=2, dim=1).values
                ssl_top1_sum += float(top2_all[:, 0].mean().item())
                ssl_margin_sum += float((top2_all[:, 0] - top2_all[:, 1]).mean().item())
                ssl_stat_batches += 1

            with torch.no_grad():
                if ENABLE_LABEL_DISTILL:
                    _update_proto_label_hist(idx_upd.detach(), targets.detach())

            if epoch >= PROTO_WARMUP_EPOCHS and LAMBDA_SSL > 0.0:
                top2 = sims.topk(k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                mask = (margin > SSL_MARGIN) & (top2[:, 0] > SSL_MIN_SIM)
                if mask.any():
                    ssl_loss = F.cross_entropy(sims[mask] / SSL_TEMP, idx_upd[mask])
                    keep_ratio = float(mask.float().mean().item())

        energy_loss = torch.tensor(0.0, device=device)
        energy_val = 0.0
        if isinstance(aux, dict) and aux.get('layer_fr', None) is not None:
            fr = aux['layer_fr']
            energy = fr[0] * float(args.fc[0]) + fr[1] * float(args.fc[1]) + fr[2] * float(args.fc[2])
            if ENERGY_TARGET > 0.0:
                energy_loss = F.relu(energy - ENERGY_TARGET)
            else:
                energy_loss = energy
            energy_val = float(energy.detach().item())

        loss = ce_loss + LAMBDA_SSL * ssl_loss + LAMBDA_ENERGY * energy_loss
        loss.backward()
        optimizer.step()
        if net_ema is not None:
            _ema_update(net_ema, net, EMA_MOMENTUM)

        if proto_bank is not None and emb is not None and idx_upd is not None:
            with torch.no_grad():
                proto_bank.update(emb.detach(), idx_upd.detach())

        loss_sum += float(loss.detach().item())
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

        ssl_loss_sum += float(ssl_loss.detach().item())
        ssl_keep_sum += float(keep_ratio)
        ssl_batches += 1

        energy_sum += float(energy_val)
        energy_loss_sum += float(energy_loss.detach().item())
        energy_batches += 1

    mean_loss = loss_sum / max(len(trainloader), 1)
    loss_train_record.append(mean_loss)

    ssl_loss_record.append(ssl_loss_sum / max(ssl_batches, 1))
    ssl_keep_ratio_record.append(ssl_keep_sum / max(ssl_batches, 1))

    if ssl_stat_batches > 0:
        ssl_top1_sim_record.append(ssl_top1_sum / ssl_stat_batches)
        ssl_margin_mean_record.append(ssl_margin_sum / ssl_stat_batches)
    else:
        ssl_top1_sim_record.append(float('nan'))
        ssl_margin_mean_record.append(float('nan'))

    if proto_bank is not None and epoch >= PROTO_WARMUP_EPOCHS:
        st = proto_bank.stats()
        proto_stats_record.append({'epoch': epoch + 1, 'active': st.active, 'usage_entropy': st.usage_entropy})

    energy_epoch_record.append({
        'epoch': epoch + 1,
        'mean_energy': energy_sum / max(energy_batches, 1),
        'mean_energy_loss': energy_loss_sum / max(energy_batches, 1),
        'lambda_energy': float(LAMBDA_ENERGY),
        'energy_target': float(ENERGY_TARGET),
    })

    dt = time.time() - t0
    fr_msg = ""
    if energy_batches > 0:
        fr_msg = f" | energy={energy_epoch_record[-1]['mean_energy']:.4f}"
    print(f"[Train] epoch={epoch+1} loss={mean_loss:.4f} {fr_msg} time={dt:.2f}s")


@torch.no_grad()
def evaluate(loader) -> Tuple[float, list, float, float]:
    net.eval()
    correct = 0
    total = 0
    fr_sum = torch.zeros(3, device=device)
    fr_batches = 0
    loss_sum = 0.0
    energy_sum = 0.0
    energy_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs, aux = net(inputs)
        loss_sum += float(criterion(outputs, targets).item())
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

        fr = None
        if isinstance(aux, dict):
            fr = aux.get('layer_fr_detached', None)
            if fr is None and aux.get('layer_fr', None) is not None:
                fr = aux['layer_fr'].detach()
        if fr is not None:
            fr_sum += fr
            fr_batches += 1
            e = fr[0] * float(args.fc[0]) + fr[1] * float(args.fc[1]) + fr[2] * float(args.fc[2])
            energy_sum += float(e.item())
            energy_batches += 1

    mean_loss = loss_sum / max(len(loader), 1)
    mean_energy = energy_sum / max(energy_batches, 1) if energy_batches > 0 else 0.0
    return _, _, mean_loss, mean_energy


def SWR_replay(epoch: int):

    if not (ENABLE_REPLAY and denoiser is not None and replay_optimizer is not None and proto_bank is not None):
        return None
    if epoch < REPLAY_WARMUP_EPOCHS:
        return None
    if REPLAY_TRIGGER_MODE == 'fixed':
        if REPLAY_EVERY_EPOCHS <= 0 or ((epoch + 1) % REPLAY_EVERY_EPOCHS) != 0:
            return None

    denoiser.train()
    net.train()

    K = max(int(REPLAY_K), 1)
    mix = float(REPLAY_CHAIN_MIX)

    denoise_losses = []
    distill_losses = []
    total_losses = []

    cos_start_list, cos_mid_list, cos_end_list = [], [], []

    spike_keep_list = []
    spikes_per_sample_list = []
    spike_rate_list = []
    spike_budget_loss_list = []
    proj_reg_list = []
    forwards = 0

    budget = float(REPLAY_SPIKE_BUDGET)
    if budget <= 0.0:
        budget = 0.05 * float(getattr(denoiser, "h", 1)) * float(getattr(denoiser, "Tb", 1))

    for _ in range(REPLAY_STEPS):
        k_idx = _sample_proto_ids_by_usage(REPLAY_BATCH)
        if k_idx is None:
            break

        with torch.no_grad():
            proto = proto_bank.prototypes[k_idx]
            proto = F.normalize(proto, p=2, dim=1, eps=1e-12)

        z = proto + sigma_schedule[0] * torch.randn_like(proto)
        z = F.normalize(z, p=2, dim=1, eps=1e-12)

        with torch.no_grad():
            cos_start_list.append(float((z * proto).sum(dim=1).mean().item()))

        denoise_loss_acc = torch.tensor(0.0, device=device)
        z_mid = None

        for t in range(K):
            t_idx = torch.full((z.size(0),), t, dtype=torch.long, device=device)
            x0_hat, spikes_per_sample, spike_keep, spike_rate_mean = denoiser(z, t_idx, return_stats=True)
            forwards += 1

            spike_keep_list.append(float(spike_keep.detach().item()))
            spikes_per_sample_list.append(float(spikes_per_sample.mean().detach().item()))
            spike_rate_list.append(float(spike_rate_mean.detach().item()))

            budget_loss = torch.relu(spikes_per_sample.mean() - budget) / max(budget, 1e-6)
            spike_budget_loss_list.append(float(budget_loss.detach().item()))

            x0_hat = F.normalize(x0_hat, p=2, dim=1, eps=1e-12)
            z = F.normalize((1.0 - mix) * z + mix * x0_hat, p=2, dim=1, eps=1e-12)

            step_loss = 1.0 - (x0_hat * proto).sum(dim=1).mean()
            if REPLAY_INTERMEDIATE_LOSS:
                denoise_loss_acc = denoise_loss_acc + step_loss

            if t == (K // 2):
                z_mid = z.detach()

        if not REPLAY_INTERMEDIATE_LOSS:
            denoise_loss_acc = 1.0 - (z * proto).sum(dim=1).mean()

        with torch.no_grad():
            if z_mid is not None:
                cos_mid_list.append(float((z_mid * proto).sum(dim=1).mean().item()))
            else:
                cos_mid_list.append(float('nan'))
            cos_end_list.append(float((z.detach() * proto).sum(dim=1).mean().item()))

        distill_loss = torch.tensor(0.0, device=device)
        if LAMBDA_REPLAY_DISTILL > 0.0 and REPLAY_DISTILL_MODE != 'none':
            if (REPLAY_DISTILL_MODE == 'ema') and (net_ema is not None):
                with torch.no_grad():
                    t_in = proto
                    if hasattr(net_ema, 'emb_proj'):
                        t_in = net_ema.emb_proj(t_in)
                    t_logits = net_ema.fc4(t_in) / max(REPLAY_DISTILL_TEMP, 1e-8)
                    t_prob = F.softmax(t_logits, dim=1).clamp_min(1e-12)

                s_in = z
                if hasattr(net, 'emb_proj'):
                    s_in = net.emb_proj(s_in)
                s_logits = net.fc4(s_in) / max(REPLAY_DISTILL_TEMP, 1e-8)
                s_logp = F.log_softmax(s_logits, dim=1)
                distill_loss = F.kl_div(s_logp, t_prob, reduction='batchmean') * (REPLAY_DISTILL_TEMP ** 2)

            elif (REPLAY_DISTILL_MODE == 'labels') and (proto_label_counts is not None):
                with torch.no_grad():
                    y = proto_label_counts[k_idx]
                    y_sum = y.sum(dim=1, keepdim=True)
                    mask = (y_sum.squeeze(1) > 0)
                if mask.any():
                    y_soft = (y[mask] / y_sum[mask]).clamp_min(1e-12)
                    z_cls = z[mask]
                    if hasattr(net, 'emb_proj'):
                        z_cls = net.emb_proj(z_cls)
                    logits = net.fc4(z_cls)
                    logp = F.log_softmax(logits, dim=1)
                    distill_loss = F.kl_div(logp, y_soft, reduction='batchmean')

        total = LAMBDA_REPLAY_DENOISE * denoise_loss_acc + LAMBDA_REPLAY_DISTILL * distill_loss

        if LAMBDA_REPLAY_SPIKE_BUDGET > 0.0:
            total = total + LAMBDA_REPLAY_SPIKE_BUDGET * budget_loss

        proj_reg = torch.tensor(0.0, device=device)
        if (REPLAY_UPDATE_EMB_PROJ and LAMBDA_REPLAY_EMB_PROJ_L2 > 0.0 and hasattr(net, 'emb_proj') and emb_proj_eye is not None):
            proj_reg = (net.emb_proj.weight - emb_proj_eye).pow(2).mean()
            total = total + LAMBDA_REPLAY_EMB_PROJ_L2 * proj_reg
        proj_reg_list.append(float(proj_reg.detach().item()))

        replay_optimizer.zero_grad()
        total.backward()
        replay_optimizer.step()

        denoise_losses.append(float(denoise_loss_acc.detach().item()))
        distill_losses.append(float(distill_loss.detach().item()))
        total_losses.append(float(total.detach().item()))

    if len(total_losses) == 0:
        return None

    denoise_mean = sum(denoise_losses) / len(denoise_losses)
    distill_mean = sum(distill_losses) / len(distill_losses)
    total_mean = sum(total_losses) / len(total_losses)

    replay_denoise_record.append(denoise_mean)
    replay_distill_record.append(distill_mean)
    replay_total_record.append(total_mean)

    cos_start = sum(cos_start_list) / max(len(cos_start_list), 1)
    cos_mid = sum(cos_mid_list) / max(len(cos_mid_list), 1)
    cos_end = sum(cos_end_list) / max(len(cos_end_list), 1)
    replay_cos_progress_record.append([cos_start, cos_mid, cos_end])

    spike_keep = sum(spike_keep_list) / max(len(spike_keep_list), 1)
    spikes_per_sample = sum(spikes_per_sample_list) / max(len(spikes_per_sample_list), 1)
    spike_rate = sum(spike_rate_list) / max(len(spike_rate_list), 1)
    spike_budget_loss = sum(spike_budget_loss_list) / max(len(spike_budget_loss_list), 1)
    proj_reg_mean = sum(proj_reg_list) / max(len(proj_reg_list), 1)

    replay_relu_keep_record.append(float(spike_keep))
    replay_active_units_record.append(float(spikes_per_sample * float(REPLAY_BATCH) * float(forwards)))  # rough total spikes proxy
    replay_forwards_record.append(int(forwards))
    replay_emb_proj_reg_record.append(float(proj_reg_mean))

    replay_spikes_per_sample_record.append(float(spikes_per_sample))
    replay_spike_rate_record.append(float(spike_rate))
    replay_spike_budget_loss_record.append(float(spike_budget_loss))

    print(f"[Replay-SWR] epoch={epoch+1} denoise={denoise_mean:.4f} distill={distill_mean:.4f} total={total_mean:.4f} | "
          f"cos(s/m/e)=({cos_start:.3f}/{cos_mid:.3f}/{cos_end:.3f}) | "
          f"spikes/sample={spikes_per_sample:.1f} spike_rate={spike_rate:.4f} keep={spike_keep:.3f} "
          f"budget={budget:.1f} budget_loss={spike_budget_loss:.3f} proj_reg={proj_reg_mean:.2e}")
    return total_mean



_last_replay_epoch = -10**9

def maybe_SWR_replay(epoch: int):

    global _last_replay_epoch

    triggered = False
    reason = "skip"
    score = 0.0

    if not ENABLE_REPLAY:
        reason = "replay_disabled"
        replay_triggered_record.append(False)
        replay_trigger_reason_record.append(reason)
        replay_trigger_score_record.append(float(score))
        return None

    if epoch < REPLAY_WARMUP_EPOCHS:
        reason = "warmup"
        replay_triggered_record.append(False)
        replay_trigger_reason_record.append(reason)
        replay_trigger_score_record.append(float(score))
        return None

    mode = str(REPLAY_TRIGGER_MODE).lower()

    if mode == 'fixed':
        out = SWR_replay(epoch)
        triggered = (out is not None)
        reason = "fixed" if triggered else "fixed_skip"
        replay_triggered_record.append(bool(triggered))
        replay_trigger_reason_record.append(reason)
        replay_trigger_score_record.append(float(score))
        return out

    if mode == 'off':
        reason = "off"
        replay_triggered_record.append(False)
        replay_trigger_reason_record.append(reason)
        replay_trigger_score_record.append(float(score))
        return None


    if (epoch - _last_replay_epoch) <= int(REPLAY_TRIGGER_COOLDOWN):
        reason = "cooldown"
        replay_triggered_record.append(False)
        replay_trigger_reason_record.append(reason)
        replay_trigger_score_record.append(float(score))
        return None

    ssl_keep = float(ssl_keep_ratio_record[-1]) if len(ssl_keep_ratio_record) > 0 else float('nan')
    top1_sim = float(ssl_top1_sim_record[-1]) if len(ssl_top1_sim_record) > 0 else float('nan')
    margin = float(ssl_margin_mean_record[-1]) if len(ssl_margin_mean_record) > 0 else float('nan')

    active_frac = 1.0
    if len(proto_stats_record) > 0:
        st = proto_stats_record[-1]
        try:
            active_frac = float(st.get('active', 0)) / max(float(NUM_PROTOTYPES), 1.0)
        except Exception:
            active_frac = 1.0

    cond_keep = (not math.isnan(ssl_keep)) and (ssl_keep < float(REPLAY_TRIGGER_SSL_KEEP_LT))
    cond_top1 = (not math.isnan(top1_sim)) and (top1_sim < float(REPLAY_TRIGGER_TOP1_SIM_LT))
    cond_margin = (not math.isnan(margin)) and (margin < float(REPLAY_TRIGGER_MARGIN_LT))
    cond_active = (active_frac < float(REPLAY_TRIGGER_ACTIVE_LT))

    score = 0.0
    if not math.isnan(ssl_keep):
        score = max(score, 1.0 - ssl_keep)
    if not math.isnan(top1_sim):
        score = max(score, max(0.0, float(REPLAY_TRIGGER_TOP1_SIM_LT) - top1_sim))
    if not math.isnan(margin):
        score = max(score, max(0.0, float(REPLAY_TRIGGER_MARGIN_LT) - margin))
    score = max(score, max(0.0, float(REPLAY_TRIGGER_ACTIVE_LT) - active_frac))

    if (epoch - _last_replay_epoch) >= int(REPLAY_TRIGGER_MAX_GAP):
        triggered = True
        reason = "max_gap"
    elif cond_keep or cond_top1 or cond_margin or cond_active:
        triggered = True
        reasons = []
        if cond_keep: reasons.append("keep")
        if cond_top1: reasons.append("top1")
        if cond_margin: reasons.append("margin")
        if cond_active: reasons.append("active")
        reason = "+".join(reasons) if len(reasons) > 0 else "trigger"
    else:
        triggered = False
        reason = "stable"

    if not triggered:
        replay_triggered_record.append(False)
        replay_trigger_reason_record.append(reason)
        replay_trigger_score_record.append(float(score))
        return None

    out = SWR_replay(epoch)
    if out is not None:
        _last_replay_epoch = epoch
        replay_triggered_record.append(True)
        replay_trigger_reason_record.append(reason)
        replay_trigger_score_record.append(float(score))
        if REPLAY_TRIGGER_VERBOSE:
            print(f"[Replay-Trigger] epoch={epoch+1} TRIGGERED reason={reason} score={score:.3f} (keep={ssl_keep:.3f}, top1={top1_sim:.3f}, margin={margin:.3f}, active_frac={active_frac:.3f})")
    else:
        replay_triggered_record.append(False)
        replay_trigger_reason_record.append("sleep_skip")
        replay_trigger_score_record.append(float(score))
    return out

def build_loaders(perm_tensor: torch.Tensor):
    ds_tr = MNIST(mnist_train, perm_tensor, in_size=args.in_size)
    ds_te = MNIST(mnist_val, perm_tensor, in_size=args.in_size)
    tr_loader = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    te_loader = torch.utils.data.DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return tr_loader, te_loader


if __name__ == "__main__":
    NUM_TASKS = _env_int("NUM_TASKS", 5)
    TASK_EPOCHS = _env_int("TASK_EPOCHS", 5)
    PERM_SEED0 = _env_int("PERM_SEED0", 1234)
    SAVE_BEST = _env_bool("SAVE_BEST", False)

    perms = _make_permutations(NUM_TASKS, seed0=PERM_SEED0)

    energy_matrix = [[None for _ in range(NUM_TASKS)] for _ in range(NUM_TASKS)]

    global_epoch = 0

    for task_id in range(NUM_TASKS):
        print("\n" + "=" * 70)
        print(f"[Task {task_id+1}/{NUM_TASKS}] perm seed = {PERM_SEED0 + task_id} (in_size={args.in_size})")
        print("=" * 70)

        trainloader, testloader = build_loaders(perms[task_id])

        for local_epoch in range(TASK_EPOCHS):
            global_epoch += 1
            epoch_idx = task_id * TASK_EPOCHS + local_epoch
            train_one_epoch(trainloader, epoch_idx)

            _, _, _, energy_eval = evaluate(testloader)
            print(f"[Eval] task={task_id} epoch={local_epoch+1}/{TASK_EPOCHS} energy={energy_eval:.4f}")


            maybe_SWR_replay(epoch_idx)

            scheduler.step()

        for eval_task in range(task_id + 1):
            _, eval_loader = build_loaders(perms[eval_task])
            _, _, _, energy_eval = evaluate(eval_loader)
            energy_matrix[task_id][eval_task] = float(energy_eval)
            print(f"[Eval after task {task_id}] on task {eval_task}: energy={energy_eval:.4f}")


