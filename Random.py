# ======================================================
# RL-adaptive CSMA vs Static CSMA (fast-learning reward)
# ======================================================

import os, random, math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------
# Reproducibility
# ----------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------
# Simulation Constants
# ----------------------
NUM_NODES      = 16
EPISODES       = 300          # slightly lower for faster runs
STEPS          = 250          # slots per episode
SLOT_TIME_S    = 1e-6         # 1 µs/slot, used for throughput units
PACKET_SIZE    = 1024         # bits
LAMBDA_RATE    = 0.35         # pkt/slot per node (Poisson arrivals)

# CW (contention window) limits
CW_MIN, CW_MAX = 8, 256
CW_INIT        = 32

# Energy model (kept, but NOT in the reward for faster learning)
TX_ENERGY_PER_BIT  = 1e-6     # J/bit
IDLE_LISTEN_EN     = 2e-7     # J per waiting node per slot
CS_BASE_EN         = 5e-8     # J per node per slot (carrier sense)
COLLISION_OVERHEAD = 6e-6     # J per collided attempt

# ----------------------
# DQN Hyperparameters
# ----------------------
GAMMA       = 0.99
LR          = 5e-4            # gentler than 1e-3
BATCH_SIZE  = 256             # larger batch for stabler learning
REPLAY_SIZE = 50_000
MIN_REPLAY  = 5_000           # learn after more diverse data
TARGET_SYNC = 200             # more frequent target sync
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 20_000

# Action space: CW multipliers (global)
CW_MULTS = np.array([0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)

# ======================================================
# Utilities
# ======================================================
def clamp_round_cw(cw):
    cw = int(round(cw))
    return max(CW_MIN, min(CW_MAX, cw))

# ======================================================
# Environment
# ======================================================
class CSMADQNEnv:
    """Global agent chooses a CW multiplier each slot (applied to all nodes)."""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.reset()

    def _reset_poisson_timers(self):
        lam = max(1e-6, LAMBDA_RATE)
        self.next_arrival_time = np.random.exponential(1/lam, self.num_nodes)

    def reset(self):
        self.time = 0
        self.cw = CW_INIT
        self.queue = np.random.randint(0, 2, self.num_nodes).astype(np.int32)
        self.success_count = 0
        self.collision_count = 0
        self.energy_used = 0.0
        self.total_delay = 0.0
        self.last_success_slot = 0
        self._reset_poisson_timers()
        return self._get_state()

    def _get_state(self):
        q_size = float(self.queue.sum()) / self.num_nodes                 # [0,1]
        succ   = float(self.success_count)  / max(1.0, self.time)
        coll   = float(self.collision_count)/ max(1.0, self.time)
        cw_n   = (self.cw - CW_MIN) / (CW_MAX - CW_MIN)                   # [0,1]
        since  = (self.time - self.last_success_slot) / max(1.0, self.time)
        return np.array([q_size, succ, coll, cw_n, since], dtype=np.float32)

    def step(self, action_multiplier):
        self.time += 1

        # Update CW from action
        self.cw = clamp_round_cw(self.cw * float(action_multiplier))

        # Nodes with a packet attempt with prob ≈ 1/CW
        p_tx = 1.0 / (self.cw + 1e-12)
        attempts_mask = (np.random.rand(self.num_nodes) < p_tx).astype(np.int32)
        attempts_mask = attempts_mask * self.queue
        n_tx = int(attempts_mask.sum())

        # Channel outcome
        if n_tx == 1:
            self.success_count += 1
            tx_idx = int(np.where(attempts_mask == 1)[0][0])
            self.queue[tx_idx] = 0
            self.last_success_slot = self.time
        elif n_tx > 1:
            self.collision_count += 1

        # Poisson arrivals
        for i in range(self.num_nodes):
            self.next_arrival_time[i] -= 1
            if self.next_arrival_time[i] <= 0:
                self.queue[i] = 1
                lam = max(1e-6, LAMBDA_RATE)
                self.next_arrival_time[i] = np.random.exponential(1/lam)

        # Energy bookkeeping (not part of reward here)
        energy = 0.0
        energy += self.num_nodes * CS_BASE_EN
        energy += n_tx * PACKET_SIZE * TX_ENERGY_PER_BIT
        if n_tx > 1:
            energy += n_tx * COLLISION_OVERHEAD
        waiters = np.sum((self.queue == 1) & (attempts_mask == 0))
        energy += waiters * IDLE_LISTEN_EN
        self.energy_used += energy

        # Delay proxy (queue length)
        self.total_delay += float(self.queue.sum())

        # ---------- SIMPLE, STRONG REWARD (fast learning) ----------
        succ = 1.0 if n_tx == 1 else 0.0
        coll = 1.0 if n_tx > 1 else 0.0
        idle = 1.0 if (n_tx == 0 and self.queue.any()) else 0.0
        queue_norm = float(self.queue.sum()) / max(1, self.num_nodes)

        W_SUCC, W_COLL, W_QLEN, W_IDLE = 1.2, 0.9, 0.6, 0.2
        reward = (
            + W_SUCC * succ
            - W_COLL * coll
            - W_QLEN * queue_norm
            - W_IDLE * idle
        )
        # -----------------------------------------------------------

        next_state = self._get_state()
        done = False
        return next_state, float(reward), done

    # KPIs per episode
    def episode_kpis(self):
        # throughput in Gb/s using SLOT_TIME_S and PACKET_SIZE
        bits = self.success_count * PACKET_SIZE
        episode_time_s = STEPS * SLOT_TIME_S
        throughput_gbps = (bits / episode_time_s) / 1e9

        latency_avg_ms  = (self.total_delay / (STEPS * self.num_nodes)) * (SLOT_TIME_S * 1e3)
        energy_per_bit  = self.energy_used / (bits + 1e-9)
        utilization_pct = (self.success_count / STEPS) * 100.0
        return throughput_gbps, latency_avg_ms, energy_per_bit, utilization_pct

# ======================================================
# DQN pieces
# ======================================================
Transition = namedtuple("Transition", ("s", "a", "r", "sp"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s  = torch.tensor(np.stack([t.s for t in batch]), dtype=torch.float32)
        a  = torch.tensor([t.a for t in batch], dtype=torch.long).unsqueeze(1)
        r  = torch.tensor([t.r for t in batch], dtype=torch.float32).unsqueeze(1)
        sp = torch.tensor(np.stack([t.sp for t in batch]), dtype=torch.float32)
        return s, a, r, sp
    def __len__(self):
        return len(self.buf)

class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# ======================================================
# Train DQN agent
# ======================================================
def dqn_train():
    env = CSMADQNEnv(NUM_NODES)
    state_dim = 5
    n_actions = len(CW_MULTS)

    q_net = QNet(state_dim, n_actions)
    tgt_net = QNet(state_dim, n_actions)
    tgt_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    rb = ReplayBuffer(REPLAY_SIZE)

    eps = EPS_START
    eps_step = 0

    returns = []
    thr_list, lat_list, en_list, util_list = [], [], [], []
    cw_trace = []

    global_step = 0

    for ep in range(EPISODES):
        s = env.reset()
        ep_return = 0.0

        for _ in range(STEPS):
            # epsilon-greedy
            if np.random.rand() < eps:
                a_idx = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    qs = q_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                    a_idx = int(torch.argmax(qs, dim=1).item())

            a_mult = CW_MULTS[a_idx]
            sp, r, _ = env.step(a_mult)
            rb.push(s, a_idx, r, sp)
            s = sp
            ep_return += r
            cw_trace.append(env.cw)
            global_step += 1

            # learn
            if len(rb) >= MIN_REPLAY:
                batch_s, batch_a, batch_r, batch_sp = rb.sample(BATCH_SIZE)
                with torch.no_grad():
                    max_q_next = tgt_net(batch_sp).max(dim=1, keepdim=True)[0]
                    target_q = batch_r + GAMMA * max_q_next
                q_vals = q_net(batch_s).gather(1, batch_a)
                loss = loss_fn(q_vals, target_q)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                # target sync
                if global_step % TARGET_SYNC == 0:
                    tgt_net.load_state_dict(q_net.state_dict())

            # eps decay
            if eps > EPS_END:
                eps_step += 1
                eps = EPS_END + (EPS_START - EPS_END) * math.exp(-eps_step / EPS_DECAY)

        # episode KPIs
        returns.append(ep_return)
        thr, lat, en, ut = env.episode_kpis()
        thr_list.append(thr); lat_list.append(lat); en_list.append(en); util_list.append(ut)

    return returns, thr_list, lat_list, en_list, util_list, cw_trace

# ======================================================
# Static baseline (fixed CW)
# ======================================================
def run_static_baseline(cw_fixed=64):
    env = CSMADQNEnv(NUM_NODES)
    thr_s, lat_s, en_s, ut_s = [], [], [], []
    for _ in range(EPISODES):
        env.reset(); env.cw = cw_fixed
        for _ in range(STEPS):
            _sp, _r, _done = env.step(1.0)   # keep CW fixed
            env.cw = cw_fixed                # force fixed after step
        thr, lat, en, ut = env.episode_kpis()
        thr_s.append(thr); lat_s.append(lat); en_s.append(en); ut_s.append(ut)
    return thr_s, lat_s, en_s, ut_s, float(np.mean(thr_s)), float(np.mean(lat_s)), float(np.mean(en_s)), float(np.mean(ut_s))

# ======================================================
# Run
# ======================================================
returns, thr_ad, lat_ad, en_ad, ut_ad, cw_trace = dqn_train()
thr_st, lat_st, en_st, ut_st, thr_s_avg, lat_s_avg, en_s_avg, ut_s_avg = run_static_baseline(cw_fixed=64)

# ======================================================
# Plots (learning curves with units)
# ======================================================
def moving_average(x, k=15):
    x = np.asarray(x, dtype=float)
    if len(x) < k: return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k

plt.figure(figsize=(12,8))

# Episode return
plt.subplot(2,2,1)
plt.plot(returns, label="Return (raw)", color="C0", alpha=0.4)
ma = moving_average(returns, 15)
if len(ma)>0:
    plt.plot(np.arange(len(ma))+15-1, ma, label="Return (MA15)", color="C1", linewidth=2)
plt.title("Episode Return (↑)")
plt.xlabel("Episode"); plt.ylabel("Return"); plt.grid(True); plt.legend()

# Throughput
plt.subplot(2,2,2)
plt.plot(thr_ad, label="Adaptive CSMA (DQN)")
plt.plot(thr_st, "r--", label=f"Static CSMA (CW=64), avg={thr_s_avg:.3f} Gb/s")
plt.title("Throughput (Gb/s) ↑"); plt.xlabel("Episode"); plt.ylabel("Throughput(Gb/s)")
plt.grid(True); plt.legend()

# Latency
plt.subplot(2,2,3)
plt.plot(lat_ad, label="Adaptive CSMA (DQN)")
plt.plot(lat_st, "r--", label=f"Static CSMA (CW=64), avg={lat_s_avg*1e3:.3f} µs", alpha=0.8)  # note: lat_s_avg is ms
plt.title("Latency (ms) ↓ (proxy via avg queue)")
plt.xlabel("Episode"); plt.ylabel("Latency(ms)"); plt.grid(True); plt.legend()

# Energy/bit
plt.subplot(2,2,4)
plt.plot(en_ad, label="Adaptive CSMA (DQN)")
plt.plot(en_st, "r--", label=f"Static CSMA (CW=64), avg={en_s_avg:.2e} J/bit")
plt.title("Energy per bit (J/bit) ↓"); plt.xlabel("Episode"); plt.ylabel("Energy/bit(J/bit)")
plt.grid(True); plt.legend()

plt.tight_layout(); plt.show()

# CW trace (optional sanity check)
plt.figure(figsize=(7,3))
plt.plot(cw_trace, alpha=0.7)
plt.title("CW over time (should settle in a sensible band)")
plt.xlabel("Step"); plt.ylabel("CW")
plt.tight_layout(); plt.show()

# ======================================================
# Separate KPI bar charts (averages, units)
# ======================================================
def avg_last(arr, k=50):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr[-k:])) if len(arr) >= k else float(np.mean(arr))

thr_ad_avg = avg_last(thr_ad)   # Gb/s
lat_ad_avg = avg_last(lat_ad)   # ms
en_ad_avg  = avg_last(en_ad)    # J/bit
ut_ad_avg  = avg_last(ut_ad)    # %

# Throughput
plt.figure(figsize=(6,4))
plt.bar(["Static CSMA","Adaptive CSMA"], [thr_s_avg, thr_ad_avg], color=["salmon","cornflowerblue"])
#for i,v in enumerate([thr_s_avg, thr_ad_avg]):
    #plt.text(i, v*1.01, f"{v:.3f}", ha="center")
plt.title("Throughput (Gb/s) — average"); plt.ylabel("Throughput(Gb/s)"); plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.show()

# Latency
plt.figure(figsize=(6,4))
plt.bar(["Static CSMA","Adaptive CSMA"], [lat_s_avg, lat_ad_avg], color=["salmon","cornflowerblue"])
#for i,v in enumerate([lat_s_avg, lat_ad_avg]):
    #plt.text(i, v*1.01, f"{v:.3f}", ha="center")
plt.title("Latency (ms) — average"); plt.ylabel("Latency(ms)"); plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.show()

# Energy per bit
plt.figure(figsize=(6,4))
plt.bar(["Static CSMA","Adaptive CSMA"], [en_s_avg, en_ad_avg], color=["salmon","cornflowerblue"])
#for i,v in enumerate([en_s_avg, en_ad_avg]):
    #plt.text(i, v*1.01, f"{v:.2e}", ha="center")
plt.title("Energy (J/bit) — average"); plt.ylabel("Energy/bit(J/bit)"); plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.show()

# Utilization
plt.figure(figsize=(6,4))
plt.bar(["Static CSMA","Adaptive CSMA"], [ut_s_avg, ut_ad_avg], color=["salmon","cornflowerblue"])
#for i,v in enumerate([ut_s_avg, ut_ad_avg]):
   # plt.text(i, v*1.01, f"{v:.2f}%", ha="center")
plt.title("Utilization (%) — average"); plt.ylabel("channel_utilization (%)"); plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
