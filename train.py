"""
Training for AeroMind (from-scratch tabular RL).

    python train.py                       # trains all three RL configurations
    python train.py --config full --episodes 3000

Configurations
    iql    : K independent-Q searchers only (no coordinator)
    coord  : searchers + Expected-SARSA coordinator
    full   : searchers + coordinator + relay UAV
"""
import argparse, os, pickle, time
import numpy as np
import pandas as pd
from env import Config, SearchEnv
from agents import RLTeam, run_episode

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
LOG_DIR = os.path.join(HERE, "logs")

CONFIGS = {
    "iql": dict(use_coord=False, use_relay=False, label="IQL searchers"),
    "coord": dict(use_coord=True, use_relay=False, label="IQL + coordinator"),
    "full": dict(use_coord=True, use_relay=True, label="IQL + coordinator + relay"),
}


def make_cfg(config_name, **over):
    cfg = Config(use_relay=CONFIGS[config_name]["use_relay"])
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def train(config_name="coord", episodes=2000, alpha=0.15, gamma=0.9, eps_start=1.0, eps_end=0.05,
          seed=0, progress=None, cfg_over=None, share_tables=False):
    """Returns (team, log DataFrame).  progress(ep, total, row_dict) is called every episode."""
    spec = CONFIGS[config_name]
    cfg = make_cfg(config_name, **(cfg_over or {}))
    env = SearchEnv(cfg)
    team = RLTeam(cfg, use_coord=spec["use_coord"], use_relay=spec["use_relay"],
                  alpha=alpha, gamma=gamma, seed=seed, share_tables=share_tables)
    decay = (eps_end / eps_start) ** (1.0 / max(1, episodes * 0.8))     # reach eps_end at 80% of training
    eps = eps_start
    rows = []
    for ep in range(episodes):
        team.set_eps(eps)
        m = run_episode(env, team, seed=10_000 * (seed + 1) + ep, learn=True)
        m["episode"] = ep
        m["epsilon"] = eps
        m["q_states"] = sum(len(s.Q) for s in team.searchers) if not share_tables else len(team.searchers[0].Q)
        rows.append(m)
        eps = max(eps_end, eps * decay)
        if progress:
            progress(ep + 1, episodes, m)
    team.set_eps(0.0)
    return team, pd.DataFrame(rows)


def save(team, log, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(team.to_dict(), f)
    log.to_csv(os.path.join(LOG_DIR, f"train_{name}.csv"), index=False)


def load(name, cfg=None):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    return RLTeam.from_dict(cfg or make_cfg(name), d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", choices=["all"] + list(CONFIGS))
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    names = list(CONFIGS) if a.config == "all" else [a.config]
    for n in names:
        t0 = time.time()
        def cb(ep, tot, row):
            if ep % 250 == 0:
                print(f"[{n}] ep {ep}/{tot}  eps={row['epsilon']:.2f}  success={row['success']:.0f}  "
                      f"ttl={row['time_to_locate']:.0f}  ret={row['return']:.0f}", flush=True)
        team, log = train(n, a.episodes, a.alpha, a.gamma, seed=a.seed, progress=cb)
        save(team, log, n + a.tag)
        print(f"[{n}] done in {time.time() - t0:.0f}s, last-200 success={log.success.tail(200).mean():.2f}")
