"""
Evaluation utilities: run N episodes per method on identical seeds and tabulate metrics.

    python evaluate.py                 # table over all methods + comm-loss sweep
"""
import sys
import numpy as np
import pandas as pd
from env import Config, SearchEnv
from agents import RandomTeam, GreedyTeam, run_episode
import train as T

METRICS = ["success", "found_frac", "golden_frac", "time_to_locate", "coverage",
           "redundancy", "crashes", "downs", "return"]
LABELS = {
    "random": "Random search",
    "greedy": "Greedy heuristic",
    "iql": "IQL searchers",
    "coord": "IQL + coordinator",
    "full": "IQL + coordinator + relay",
}


def get_team(key, cfg, models=None):
    """models: optional dict of in-memory trained teams (overrides files)."""
    if key == "random":
        return RandomTeam(0)
    if key == "greedy":
        return GreedyTeam(0)
    if models and key in models:
        team = models[key]
        team.cfg = cfg
        return team
    return T.load(key, cfg)


def evaluate(keys, episodes=100, comm_zones=-1, seed0=900_000, models=None, n_search=None):
    rows = []
    for key in keys:
        use_relay = key == "full"
        over = dict(comm_zones=comm_zones)
        if n_search:
            over["n_search"] = n_search
        cfg = T.make_cfg(key if key in T.CONFIGS else "iql", **over)
        cfg.use_relay = use_relay
        team = get_team(key, cfg, models)
        if team is None:
            continue
        env = SearchEnv(cfg)
        res = [run_episode(env, team, seed=seed0 + e) for e in range(episodes)]
        df = pd.DataFrame(res)
        row = {"method": LABELS[key], "key": key}
        row.update({m: float(df[m].mean()) for m in METRICS})
        row["episodes"] = episodes
        rows.append(row)
    return pd.DataFrame(rows)


def comm_sweep(keys, episodes=60, zones=(0, 1, 2, 3), models=None):
    out = []
    for z in zones:
        df = evaluate(keys, episodes, comm_zones=z, models=models)
        df["comm_zones"] = z
        out.append(df)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    keys = ["random", "greedy", "iql", "coord", "full"]
    pd.set_option("display.width", 200)
    df = evaluate(keys, episodes=200)
    print(df.drop(columns=["key"]).round(3).to_string(index=False))
    print()
    sw = comm_sweep(keys, episodes=100)
    print(sw.pivot(index="method", columns="comm_zones", values="time_to_locate").round(1))
    print(sw.pivot(index="method", columns="comm_zones", values="golden_frac").round(2))
