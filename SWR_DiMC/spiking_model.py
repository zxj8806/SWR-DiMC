import torch
import torch.nn as nn
import torch.nn.functional as F
from SWR_DiMC.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay

output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc

phase_max = args.phase_max
cycle_min = args.cycle_min
cycle_max = args.cycle_max
duty_cycle_min = args.duty_cycle_min
duty_cycle_max = args.duty_cycle_max


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (input - thresh).abs() < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


def mem_update_skip_woDecay(ops, x, mem, spike, mask):
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    mem = mem * decay * (1. - spike) + ops(x)
    mem = torch.where(mask == 0, pre_mem, mem)
    spike = act_fun(mem) * mask
    return mem, spike


class SWR_DiMC_general(nn.Module):

    def __init__(self, T=784):
        super().__init__()
        self.T = int(T)
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

        self.emb_proj = nn.Linear(cfg_fc[2], cfg_fc[2], bias=False)
        with torch.no_grad():
            if self.emb_proj.weight.shape[0] == self.emb_proj.weight.shape[1]:
                self.emb_proj.weight.copy_(torch.eye(self.emb_proj.weight.shape[0]))

        self.fc4 = nn.Linear(cfg_fc[2], output_size)

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0], duty_cycle_max[0], phase_max[0], self.T)
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1], duty_cycle_max[1], phase_max[1], self.T)
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2], duty_cycle_max[2], phase_max[2], self.T)

        self.spike_recorder = []

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=784):
        mask = []
        dc_steps = torch.linspace(min_dc, max_dc, steps=dim)
        cycles = torch.linspace(c_min, c_max, steps=dim)
        phase_shifts = torch.linspace(0, int(phase_shift_max * c_max), steps=dim)

        for cycle, dc, phase_shift in zip(cycles, dc_steps, phase_shifts):
            cycle = int(torch.round(cycle))
            on_length = int(torch.round(dc * cycle))
            off_length = cycle - on_length
            pattern = [1] * on_length + [0] * off_length

            phase_shift = int(torch.round(phase_shift))
            pattern = pattern[-phase_shift:] + pattern[:-phase_shift]

            full_pattern = pattern * (T // cycle) + pattern[:T % cycle]
            mask.append(full_pattern)

        return torch.tensor(mask, dtype=torch.float32, device=device)

    def _ensure_3d_input(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            return x.view(N, self.T, self.input_size)
        raise ValueError(f"Unexpected input dim {x.dim()}, expected 2 or 3.")

    def forward(self, x: torch.Tensor, record_spikes: bool = False):
        time_window = self.T
        N = x.size(0)

        h1_mem = torch.zeros(N, cfg_fc[0], device=device)
        h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = torch.zeros(N, cfg_fc[1], device=device)
        h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = torch.zeros(N, cfg_fc[2], device=device)
        h3_spike = torch.zeros(N, cfg_fc[2], device=device)

        output_sum = torch.zeros(N, output_size, device=device)

        h1_spike_sums = torch.zeros_like(h1_spike)
        h2_spike_sums = torch.zeros_like(h2_spike)
        h3_spike_sums = torch.zeros_like(h3_spike)

        if record_spikes:
            self.spike_recorder = []

        x = self._ensure_3d_input(x)

        for step in range(time_window):
            input_x = x[:, step, :]
            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x, h1_mem, h1_spike, self.mask1[:, step])
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, self.mask2[:, step])
            h3_mem, h3_spike = mem_update_skip_woDecay(self.fc3, h2_spike, h3_mem, h3_spike, self.mask3[:, step])

            h3_proj = self.emb_proj(h3_spike)
            out_t = self.fc4(h3_proj)
            output_sum = output_sum + out_t

            h1_spike_sums = h1_spike_sums + h1_spike
            h2_spike_sums = h2_spike_sums + h2_spike
            h3_spike_sums = h3_spike_sums + h3_spike

            if record_spikes:
                self.spike_recorder.append((h1_spike[0], h2_spike[0], h3_spike[0]))

        outputs = output_sum / float(time_window)

        fr1 = h1_spike_sums.sum() / (h1_spike_sums.numel() * float(time_window))
        fr2 = h2_spike_sums.sum() / (h2_spike_sums.numel() * float(time_window))
        fr3 = h3_spike_sums.sum() / (h3_spike_sums.numel() * float(time_window))
        layer_fr = torch.stack([fr1, fr2, fr3], dim=0)

        emb_raw = h3_spike_sums / float(time_window)
        emb = self.emb_proj(emb_raw)
        emb = F.normalize(emb, p=2, dim=1, eps=1e-12)

        aux = {
            'emb': emb,
            'emb_raw': emb_raw,
            'layer_fr': layer_fr,
            'layer_fr_detached': layer_fr.detach(),
        }
        return outputs, aux

    def collect_spikes(self):
        return self.spike_recorder
