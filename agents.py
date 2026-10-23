"""
Agents for AeroMind.  All learning is tabular and written from scratch.

  SearcherAgent   - Independent Q-Learning  (one Q-table per UAV)
  CoordinatorAgent- Expected SARSA          (one fleet-wide Q-table, coarse state)
  RelayAgent      - Q-Learning              (relay UAV that restores radio links)

plus two non-learning baselines (RandomTeam, GreedyTeam) and the RLTeam that
wires the learning agents together and runs one episode.
"""
import numpy as np
from env import (SearchEnv, Config, N_SEARCH_ACTIONS, N_RELAY_ACTIONS, N_COORD_ACTIONS,
                 COORD_PERIOD, COORD_HOLD, A_SCAN, A_RETURN, SEARCH_ACTION_NAMES,
                 RELAY_ACTION_NAMES, DIRS8)


# ============================================================ learning agents
class QAgent:
    """Tabular Q-Learning:  Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]"""

    def __init__(self, n_actions, alpha=0.15, gamma=0.9, eps=1.0, seed=0):
        self.nA, self.alpha, self.gamma, self.eps = n_actions, alpha, gamma, eps
        self.Q = {}
        self.rng = np.random.default_rng(seed)

    def q(self, s):
        v = self.Q.get(s)
        if v is None:
            v = self.Q[s] = np.zeros(self.nA)
        return v

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.nA))
        q = self.q(s)
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def update(self, s, a, r, s2, done):
        q = self.q(s)
        target = r if done else r + self.gamma * self.q(s2).max()
        q[a] += self.alpha * (target - q[a])


class SearcherAgent(QAgent):
    """UAV searcher.  Independent Q-Learning: teammates are just part of the environment."""

    def __init__(self, **kw):
        super().__init__(N_SEARCH_ACTIONS, **kw)


class RelayAgent(QAgent):
    def __init__(self, **kw):
        super().__init__(N_RELAY_ACTIONS, **kw)


class CoordinatorAgent(QAgent):
    """Fleet coordinator learning with Expected SARSA over an epsilon-greedy policy:

        Q(s,a) <- Q(s,a) + alpha [ R + gamma * sum_a' pi(a'|s') Q(s',a') - Q(s,a) ]
    """

    def __init__(self, **kw):
        super().__init__(N_COORD_ACTIONS, **kw)

    def expected_q(self, s):
        q = self.q(s)
        pi = np.full(self.nA, self.eps / self.nA)
        best = np.flatnonzero(q == q.max())
        pi[best] += (1 - self.eps) / len(best)
        return float((pi * q).sum())

    def update(self, s, a, r, s2, done):
        q = self.q(s)
        target = r if done else r + self.gamma * self.expected_q(s2)
        q[a] += self.alpha * (target - q[a])


# ================================================================= baselines
def _dir_free(env, i, d):
    r, c = env.pos[i]
    dr, dc = DIRS8[d]
    return env.valid(r + dr, c + dc)


class RandomTeam:
    name = "Random search"
    use_coord = False
    use_relay = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def begin(self, env): pass

    def decide(self, env, obs, learn=False):
        acts = [int(self.rng.integers(N_SEARCH_ACTIONS)) for _ in range(env.cfg.n_search)]
        return acts, None, {}

    def observe(self, env, *a, **k): pass
    def finish(self, env): pass


class GreedyTeam:
    """Hand-coded heuristic: follow thermal hint, scan when near, else nearest unexplored cell,
    return when the battery is critical.  No learning, no coordination."""
    name = "Greedy heuristic"
    use_coord = False
    use_relay = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def begin(self, env): pass

    def decide(self, env, obs, learn=False):
        acts = []
        for i in range(env.cfg.n_search):
            if not env.alive[i]:
                acts.append(0)
                continue
            h, g, b, cm, w, m = env.obs_parts(i)
            if b == 0 and env.pos[i] != env.base:
                acts.append(A_RETURN)
                continue
            if h >= 9 or h == 17:
                acts.append(A_SCAN)
                continue
            d = (h - 1) % 8 if 1 <= h <= 16 and h != 17 else g
            if d == 8:
                acts.append(int(self.rng.integers(8)))
                continue
            for off in (0, 1, -1, 2, -2):
                dd = (d + off) % 8
                if _dir_free(env, i, dd):
                    d = dd
                    break
            acts.append(d)
        return acts, None, {}

    def observe(self, env, *a, **k): pass
    def finish(self, env): pass


# ================================================================== RL team
class RLTeam:
    """K independent-Q searchers (+ optional coordinator, + optional relay UAV)."""

    def __init__(self, cfg: Config, use_coord=False, use_relay=False, alpha=0.15, gamma=0.9,
                 coord_alpha=0.1, seed=0, share_tables=False):
        self.cfg, self.use_coord, self.use_relay = cfg, use_coord, use_relay
        self.share = share_tables
        K = cfg.n_search
        self.searchers = [SearcherAgent(alpha=alpha, gamma=gamma, seed=seed * 100 + i) for i in range(K)]
        if share_tables:                       # optional "parameter sharing" configuration
            for s in self.searchers[1:]:
                s.Q = self.searchers[0].Q
        self.coord = CoordinatorAgent(alpha=coord_alpha, gamma=gamma, seed=seed * 100 + 50) if use_coord else None
        self.relay = RelayAgent(alpha=alpha, gamma=gamma, seed=seed * 100 + 60) if use_relay else None
        self.name = ("IQL searchers" + (" + coordinator" if use_coord else "") +
                     (" + relay UAV" if use_relay else ""))
        self.set_eps(1.0)
        self.reset_memory()

    # ---- bookkeeping
    def set_eps(self, e, coord_e=None):
        for s in self.searchers:
            s.eps = e
        if self.coord:
            self.coord.eps = e if coord_e is None else coord_e
        if self.relay:
            self.relay.eps = e

    def reset_memory(self):
        self.prev = [None] * self.cfg.n_search       # (s, a)
        self.c_prev = None                           # (s, a)
        self.c_acc = 0.0
        self.r_prev = None
        self.relay_acc = 0.0

    def begin(self, env):
        self.reset_memory()

    # ---- one decision round (simultaneous action selection)
    def decide(self, env, obs, learn=False):
        info = {}
        greedy = not learn
        # coordinator
        if self.coord and env.t % COORD_PERIOD == 0:
            cs = env.coord_state()
            if learn and self.c_prev is not None:
                self.coord.update(self.c_prev[0], self.c_prev[1], self.c_acc, cs, False)
            ca = self.coord.act(cs, greedy=greedy)
            who = env.apply_coord(ca)
            self.c_prev, self.c_acc = (cs, ca), 0.0
            info["coord"] = dict(state=cs, action=ca, searcher=who)
            # coordinator acted -> searcher observations that depend on assignment change
            obs = [env.obs(i) if env.alive[i] else None for i in range(env.cfg.n_search)]
        acts = []
        qinfo = []
        for i in range(env.cfg.n_search):
            if not env.alive[i]:
                acts.append(0)
                qinfo.append(None)
                self.prev[i] = None
                continue
            a = self.searchers[i].act(obs[i], greedy=greedy)
            acts.append(a)
            self.prev[i] = (obs[i], a)
            qinfo.append(self.searchers[i].q(obs[i]).copy())
        info["q"] = qinfo
        relay_act = None
        if self.relay:
            rs = env.relay_state()
            if learn and self.r_prev is not None:
                self.relay.update(self.r_prev[0], self.r_prev[1], self.relay_acc, rs, False)
                self.relay_acc = 0.0
            ra = self.relay.act(rs, greedy=greedy)
            self.r_prev = (rs, ra)
            relay_act = ra
            info["relay"] = dict(state=rs, action=ra)
        return acts, relay_act, info

    def observe(self, env, rew, team_r, done, next_obs, learn=False):
        """Called after env.step with next observations."""
        if self.coord:
            self.c_acc += team_r
        if self.relay:
            K_alive = max(1, sum(env.alive))
            conn = sum(1 for i in range(env.cfg.n_search) if env.alive[i] and env.comm[i])
            self.relay_acc += conn / K_alive - 0.02 * (self.r_prev[1] < 8 if self.r_prev else 0)
        if not learn:
            return
        for i in range(env.cfg.n_search):
            if self.prev[i] is None:
                continue
            s, a = self.prev[i]
            dead = not env.alive[i]
            term = done or dead
            s2 = None if term else next_obs[i]
            self.searchers[i].update(s, a, rew[i], s2, term)

    def finish(self, env):
        """Terminal updates for coordinator and relay."""
        if self.coord and self.c_prev is not None:
            self.coord.update(self.c_prev[0], self.c_prev[1], self.c_acc, None, True)
        if self.relay and self.r_prev is not None:
            self.relay.update(self.r_prev[0], self.r_prev[1], self.relay_acc, None, True)

    # ---- (de)serialisation
    def to_dict(self):
        def pack(agent):
            return {int(k): v.astype(np.float32) for k, v in agent.Q.items()}
        d = dict(use_coord=self.use_coord, use_relay=self.use_relay, share=self.share,
                 searchers=[pack(s) for s in self.searchers] if not self.share else [pack(self.searchers[0])])
        if self.coord:
            d["coord"] = pack(self.coord)
        if self.relay:
            d["relay"] = pack(self.relay)
        return d

    @classmethod
    def from_dict(cls, cfg, d):
        team = cls(cfg, use_coord=d["use_coord"], use_relay=d["use_relay"], share_tables=d.get("share", False))
        def unpack(agent, q):
            agent.Q = {int(k): np.array(v, dtype=float) for k, v in q.items()}
        if d.get("share"):
            unpack(team.searchers[0], d["searchers"][0])
            for s in team.searchers[1:]:
                s.Q = team.searchers[0].Q
        else:
            for s, q in zip(team.searchers, d["searchers"]):
                unpack(s, q)
        if team.coord:
            unpack(team.coord, d["coord"])
        if team.relay:
            unpack(team.relay, d["relay"])
        team.set_eps(0.0)
        return team


# ===================================================================== runner
def run_episode(env: SearchEnv, team, seed=None, learn=False, record=False):
    """observe -> decide -> act -> reward -> learn, repeated until the mission ends."""
    env.reset(seed)
    team.begin(env)
    K = env.cfg.n_search
    frames, total = [], np.zeros(K)
    obs = [env.obs(i) for i in range(K)]
    team_total = 0.0
    while not env.done:
        acts, relay_act, info = team.decide(env, obs, learn=learn)
        pre = env.snapshot() if record else None
        rew, team_r, done = env.step(acts, relay_act)
        next_obs = [env.obs(i) if env.alive[i] else None for i in range(K)]
        team.observe(env, rew, team_r, done, next_obs, learn=learn)
        total += rew
        team_total += team_r
        if record:
            snap = env.snapshot()
            snap.update(acts=list(acts), rew=list(rew), info=info, team_r=team_r,
                        relay_act=relay_act)
            frames.append(snap)
        obs = next_obs
    if learn:
        team.finish(env)
    m = env.metrics()
    m["return"] = float(total.sum())
    m["team_return"] = float(team_total)
    return (m, frames) if record else m
