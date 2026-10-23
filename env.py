"""
AeroMind environment: a high-altitude search-and-rescue grid world.

K searcher UAVs (plus an optional relay UAV and a coordinator) look for buried
victims.  Every step is one minute.  Survival probability decays with time,
the thermal signal of a buried victim fades, wind drifts the UAVs, batteries
are limited, cliffs are no-fly cells and some zones have no radio coverage.

Everything here is plain numpy / python.  No RL library is used anywhere.
"""
from dataclasses import dataclass, replace
from collections import deque
import numpy as np

# ---------------------------------------------------------------- constants
DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SIGN_TO_DIR = {d: i for i, d in enumerate(DIRS8)}
WIND_VECS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]      # calm, N, E, S, W
WIND_NAMES = ["calm", "N", "E", "S", "W"]

A_SCAN = 8            # hover-and-scan
A_RETURN = 9          # return-to-base / relay point
N_SEARCH_ACTIONS = 10
SEARCH_ACTION_NAMES = DIR_NAMES + ["SCAN", "RETURN"]

N_RELAY_ACTIONS = 9   # 8 moves + hold
RELAY_ACTION_NAMES = DIR_NAMES + ["HOLD"]

ZONES_PER_SIDE = 3
N_ZONES = ZONES_PER_SIDE * ZONES_PER_SIDE
N_COORD_ACTIONS = N_ZONES + 1    # assign nearest free searcher to zone z, or HOLD
COORD_HOLD = N_ZONES
COORD_PERIOD = 4                 # coordinator decides every 4 minutes
CRITICAL_BASE = 10_000_000       # state-key offset for the 'battery critical' abstraction


@dataclass
class Config:
    size: int = 15
    n_search: int = 4
    n_victims: int = 3
    max_steps: int = 200
    battery: int = 70
    p_nofly: float = 0.07
    wind_change: float = 0.05
    wind_push: float = 0.30
    comm_zones: int = -1          # -1 -> random 0..3 each episode
    relay_radius: int = 3
    base_radius: int = 2
    golden: int = 45              # "golden window" in minutes
    use_relay: bool = False


def survival(t):
    """Probability that a victim buried for t minutes is still alive."""
    return float(np.exp(-(t / 60.0) ** 2))


def signal(t):
    """Thermal / beacon signal strength after t minutes (decays)."""
    return float(np.exp(-t / 120.0))


class SearchEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        n = cfg.size
        self.n = n
        self.base = (0, 0)
        rr, cc = np.mgrid[0:n, 0:n]
        self.rr, self.cc = rr, cc
        side = n // ZONES_PER_SIDE
        self.zone_id = (rr // side) * ZONES_PER_SIDE + (cc // side)
        self.zone_id = np.minimum(self.zone_id, N_ZONES - 1)
        self.zone_masks = [self.zone_id == z for z in range(N_ZONES)]
        self.zone_centres = [
            (float(rr[m].mean()), float(cc[m].mean())) for m in self.zone_masks
        ]
        self.base_area = (np.maximum(rr, cc) <= cfg.base_radius)

    # ------------------------------------------------------------- reset
    def reset(self, seed=None):
        cfg, n = self.cfg, self.n
        self.rng = np.random.default_rng(seed)
        rng = self.rng

        # terrain: cliffs / crevasses are no-fly cells; keep everything reachable
        while True:
            nofly = rng.random((n, n)) < cfg.p_nofly
            nofly &= ~(np.maximum(self.rr, self.cc) <= cfg.base_radius + 1)
            dist = self._bfs(nofly)
            unreachable = ~np.isfinite(dist) & ~nofly
            nofly |= unreachable                       # seal unreachable pockets
            dist[nofly] = np.inf
            break
        self.nofly = nofly
        self.dist_base = dist
        self.free = ~nofly

        # communication-loss zones
        k = cfg.comm_zones if cfg.comm_zones >= 0 else int(rng.integers(0, 4))
        lost = np.zeros((n, n), bool)
        for _ in range(k):
            h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
            r0 = int(rng.integers(2, n - h + 1))
            c0 = int(rng.integers(2, n - w + 1))
            lost[r0:r0 + h, c0:c0 + w] = True
        lost &= ~self.base_area
        self.comm_lost = lost
        self.n_comm_zones = k

        # victims (buried at t=0)
        cand = np.argwhere(self.free & (dist >= 5))
        idx = rng.choice(len(cand), size=min(cfg.n_victims, len(cand)), replace=False)
        self.victims = [tuple(int(x) for x in cand[i]) for i in idx]
        self.v_found = [False] * len(self.victims)
        self.v_found_t = [None] * len(self.victims)
        self.v_found_by = [None] * len(self.victims)

        # wind
        self.wind = int(rng.integers(0, 5))

        # searchers
        K = cfg.n_search
        self.pos = [self.base for _ in range(K)]
        self.battery = [cfg.battery] * K
        self.alive = [True] * K
        self.assign = [-1] * K
        self.own_known = [np.zeros((n, n), bool) | nofly for _ in range(K)]
        self.shared_known = np.zeros((n, n), bool) | nofly
        self.covered = np.zeros((n, n), bool) | nofly      # ground truth
        for i in range(K):
            self._cover(i, self.base)

        self.relay_pos = (1, 1)
        self.pings = {}                     # victim idx -> (zone, time)
        self.t = 0
        self.done = False

        # statistics
        self.stats = dict(revisits=0, collisions=0, moves=0, crashes=0, downs=0,
                          comm_lost_steps=0, scans=0, recharges=0)
        self.comm = [True] * K
        self._update_comm()
        return self

    # ------------------------------------------------------------ helpers
    def _bfs(self, nofly):
        n = self.n
        dist = np.full((n, n), np.inf)
        dist[0, 0] = 0
        q = deque([(0, 0)])
        while q:
            r, c = q.popleft()
            for dr, dc in DIRS8:
                a, b = r + dr, c + dc
                if 0 <= a < n and 0 <= b < n and not nofly[a, b] and dist[a, b] == np.inf:
                    dist[a, b] = dist[r, c] + 1
                    q.append((a, b))
        return dist

    def valid(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n and self.free[r, c]

    def _cover(self, i, p):
        self.own_known[i][p] = True
        self.covered[p] = True

    def _update_comm(self):
        cfg = self.cfg
        for i in range(cfg.n_search):
            r, c = self.pos[i]
            ok = (not self.comm_lost[r, c]) or self.base_area[r, c]
            if not ok and cfg.use_relay:
                rr, rc = self.relay_pos
                ok = max(abs(rr - r), abs(rc - c)) <= cfg.relay_radius
            self.comm[i] = bool(ok) and self.alive[i]

    def _sync_knowledge(self):
        for i in range(self.cfg.n_search):
            if self.comm[i]:
                self.shared_known |= self.own_known[i]
        for i in range(self.cfg.n_search):
            if self.comm[i]:
                self.own_known[i] = self.shared_known.copy()

    # ------------------------------------------------- searcher observation
    def hint(self, i):
        """Thermal hint (range 2): 0 none, 1-8 direction (far), 9-16 direction (near), 17 here."""
        r, c = self.pos[i]
        best, best_read = 0, 0.10
        s = signal(self.t)
        for v, (vr, vc) in enumerate(self.victims):
            if self.v_found[v]:
                continue
            d = max(abs(vr - r), abs(vc - c))
            if d > 2:
                continue
            read = s * (1 - d / 3.0)
            if read > best_read:
                if d == 0:
                    code = 17
                else:
                    sdir = SIGN_TO_DIR[(int(np.sign(vr - r)), int(np.sign(vc - c)))]
                    code = (9 if d == 1 else 1) + sdir
                best, best_read = code, read
        return best

    def goal_dir(self, i):
        """Direction (0-7) of the nearest unexplored cell, biased to the assigned zone.  8 = none."""
        known = self.own_known[i]
        cand = ~known
        z = self.assign[i]
        if z >= 0:
            zc = cand & self.zone_masks[z]
            if zc.any():
                cand = zc
        if not cand.any():
            return 8
        r, c = self.pos[i]
        ys, xs = np.nonzero(cand)
        dr, dc = ys - r, xs - c
        d = np.maximum(np.abs(dr), np.abs(dc)) * 100 + np.abs(dr) + np.abs(dc)
        j = int(np.argmin(d))
        return SIGN_TO_DIR[(int(np.sign(dr[j])), int(np.sign(dc[j])))]

    def battery_level(self, i):
        margin = self.battery[i] - self.dist_base[self.pos[i]]
        return 0 if margin <= 8 else (1 if margin <= 20 else 2)

    def block_mask(self, i):
        """8-bit terrain window: bit d set when the neighbouring cell in direction d is blocked."""
        r, c = self.pos[i]
        m = 0
        for b, (dr, dc) in enumerate(DIRS8):
            if not self.valid(r + dr, c + dc):
                m |= 1 << b
        return m

    def obs_parts(self, i):
        return (self.hint(i), self.goal_dir(i), self.battery_level(i),
                int(self.comm[i]), self.wind, self.block_mask(i))

    def obs(self, i):
        """Integer key of the searcher's discretised local observation (the Q-table state).

        (hint, goal direction, battery level, link up?, wind calm/windy, 8-bit terrain window)
        State abstraction: when the battery is critical only the terrain window matters,
        because the only sensible behaviour is to fly home.
        """
        h, g, b, cm, w, m = self.obs_parts(i)
        if b == 0:
            return CRITICAL_BASE + m
        return ((((h * 9 + g) * 3 + (b - 1)) * 2 + cm) * 2 + (1 if w else 0)) * 256 + m

    # ------------------------------------------------- coordinator interface
    def zone_coverage(self):
        """Fraction of each zone that the *team* knows to be covered (via comms)."""
        out = []
        for m in self.zone_masks:
            free = m & self.free
            tot = free.sum()
            out.append(float((self.shared_known & free).sum() / tot) if tot else 1.0)
        return out

    def queue_zone(self):
        """Zone of the most recent unexpired ping of an unfound victim (N_ZONES if the queue is empty)."""
        qz, best_t = N_ZONES, -1
        for v, (z, t) in list(self.pings.items()):
            if self.v_found[v] or self.t - t > 20:
                self.pings.pop(v, None)
                continue
            if t > best_t:
                best_t, qz = t, z
        return qz

    def coord_state(self):
        cov = self.zone_coverage()
        least = int(np.argmin(cov))
        spread = max(cov) - min(cov)
        sb = 0 if spread < 0.25 else (1 if spread < 0.6 else 2)
        qz = self.queue_zone()
        n_conn = sum(1 for i in range(self.cfg.n_search) if self.comm[i] and self.alive[i])
        cb = 0 if n_conn == 0 else (1 if n_conn <= 2 else 2)
        return ((least * 3 + sb) * (N_ZONES + 1) + qz) * 3 + cb

    def apply_coord(self, a):
        """Reassign the best-placed reachable searcher to zone a (or HOLD)."""
        if a == COORD_HOLD:
            return None
        zc = self.zone_centres[a]
        best, best_d = None, 1e9
        cov = self.zone_coverage()
        rescan = a == self.queue_zone()
        for i in range(self.cfg.n_search):
            if not (self.alive[i] and self.comm[i]) or self.assign[i] == a:
                continue
            if self.battery_level(i) == 0:
                continue
            cz = self.assign[i]
            if cz >= 0 and cov[cz] < 0.7 and not rescan:      # don't pull a UAV out of a zone it is still sweeping
                continue
            d = max(abs(self.pos[i][0] - zc[0]), abs(self.pos[i][1] - zc[1]))
            if d < best_d:
                best, best_d = i, d
        if best is not None:
            self.assign[best] = a
        return best

    # ------------------------------------------------------- relay interface
    def relay_state(self):
        """Direction / distance to the nearest searcher that has lost its link."""
        rr, rc = self.relay_pos
        best, bd = None, 1e9
        for i in range(self.cfg.n_search):
            if self.alive[i] and not self.comm[i]:
                r, c = self.pos[i]
                d = max(abs(r - rr), abs(c - rc))
                if d < bd:
                    best, bd = (r - rr, c - rc), d
        if best is None:
            code = 0
            db = 0
        else:
            if bd == 0:
                code = 9
            else:
                code = 1 + SIGN_TO_DIR[(int(np.sign(best[0])), int(np.sign(best[1])))]
            db = 0 if bd <= 3 else (1 if bd <= 6 else 2)
        zone = int(self.zone_id[rr, rc])
        return (code * 3 + db) * N_ZONES + zone      # 10*3*9 = 270 states

    # ------------------------------------------------------------------ step
    def _return_step(self, p):
        r, c = p
        if p == self.base:
            return p
        best, bd = p, self.dist_base[r, c]
        for dr, dc in DIRS8:
            a, b = r + dr, c + dc
            if self.valid(a, b) and self.dist_base[a, b] < bd:
                best, bd = (a, b), self.dist_base[a, b]
        return best

    def step(self, acts, relay_act=None):
        """acts: list of searcher actions (ignored for dead UAVs). Returns (rewards, team_reward, done)."""
        cfg = self.cfg
        K = cfg.n_search
        rew = [0.0] * K
        team = 0.0
        order = list(self.rng.permutation(K))
        newly_cov = 0

        # relay moves first (it is always linked to base)
        if cfg.use_relay and relay_act is not None and relay_act < 8:
            dr, dc = DIRS8[relay_act]
            a, b = self.relay_pos[0] + dr, self.relay_pos[1] + dc
            if 0 <= a < self.n and 0 <= b < self.n:
                self.relay_pos = (a, b)

        for i in order:
            if not self.alive[i]:
                continue
            a = acts[i]
            r, c = self.pos[i]
            crit0 = self.battery_level(i) == 0
            d0 = self.dist_base[r, c]
            rew[i] -= 0.1                                  # energy
            self.battery[i] -= 1
            target = (r, c)
            if a < 8:
                dr, dc = DIRS8[a]
                tr, tc = r + dr, c + dc
                if self.valid(tr, tc):
                    target = (tr, tc)
                    self.stats["moves"] += 1
                else:
                    rew[i] -= 3.0                          # crash / out of bounds -> bounce
                    self.stats["crashes"] += 1
            elif a == A_RETURN:
                target = self._return_step((r, c))
                self.stats["moves"] += 1
            else:                                          # scan
                rew[i] -= 0.2
                self.stats["scans"] += 1
            # wind drift
            if self.wind and self.rng.random() < cfg.wind_push:
                wr, wc = WIND_VECS[self.wind]
                tr, tc = target[0] + wr, target[1] + wc
                if self.valid(tr, tc):
                    target = (tr, tc)
            self.pos[i] = target
            if crit0:                                      # shaping: make progress home when low on battery
                rew[i] += 0.5 * (d0 - self.dist_base[target])

            # recharge at base
            if self.base_area[target] and target == self.base:
                if self.battery[i] < cfg.battery * 0.6:
                    rew[i] += 2.0
                    self.stats["recharges"] += 1
                self.battery[i] = cfg.battery

            # coverage / revisit
            if target != self.base:
                if not self.covered[target]:
                    inzone = self.assign[i] < 0 or self.zone_id[target] == self.assign[i]
                    rew[i] += 1.0 if inzone else 0.5
                    newly_cov += 1
                else:
                    rew[i] -= 0.3
                    self.stats["revisits"] += 1
            self._cover(i, target)

            # victim detection
            s = signal(self.t)
            for v, (vr, vc) in enumerate(self.victims):
                if self.v_found[v]:
                    continue
                d = max(abs(vr - target[0]), abs(vc - target[1]))
                p = 0.0
                if a == A_SCAN and d <= 1:
                    p = 0.10 + 0.85 * s
                elif d == 0:
                    p = 0.15
                if p > 0 and self.rng.random() < p:
                    self.v_found[v] = True
                    self.v_found_t[v] = self.t
                    self.v_found_by[v] = i
                    bonus = 60.0 * survival(self.t)
                    rew[i] += bonus
                    team += bonus

            # battery death
            if self.battery[i] <= 0 and target != self.base:
                self.alive[i] = False
                rew[i] -= 20.0
                self.stats["downs"] += 1

        # overlap / collision penalty
        occ = {}
        for i in range(K):
            if self.alive[i] and self.pos[i] != self.base:
                occ.setdefault(self.pos[i], []).append(i)
        for cell, ids in occ.items():
            if len(ids) > 1:
                for i in ids:
                    rew[i] -= 2.0
                self.stats["collisions"] += len(ids)

        # environment dynamics
        if self.rng.random() < cfg.wind_change:
            self.wind = int(self.rng.integers(0, 5))
        self.t += 1
        self._update_comm()
        self.stats["comm_lost_steps"] += sum(1 for i in range(K) if self.alive[i] and not self.comm[i])
        self._sync_knowledge()

        # pings for the detection queue (only if the link is up)
        for i in range(K):
            if self.alive[i] and self.comm[i] and self.hint(i) != 0:
                for v, (vr, vc) in enumerate(self.victims):
                    if not self.v_found[v] and max(abs(vr - self.pos[i][0]), abs(vc - self.pos[i][1])) <= 2:
                        self.pings[v] = (int(self.zone_id[self.pos[i]]), self.t)

        remaining = [v for v in range(len(self.victims)) if not self.v_found[v]]
        team += 0.1 * newly_cov
        team -= 0.05 * sum(survival(self.t) for _ in remaining)

        self.done = (not remaining) or self.t >= cfg.max_steps or not any(self.alive)
        return rew, team, self.done

    # --------------------------------------------------------------- reporting
    def metrics(self):
        cfg = self.cfg
        V = len(self.victims)
        found_t = [t if t is not None else cfg.max_steps for t in self.v_found_t]
        golden = sum(1 for t in self.v_found_t if t is not None and t <= cfg.golden)
        free_cells = int(self.free.sum())
        cov = float((self.covered & self.free).sum() / free_cells)
        moves = max(1, self.stats["moves"])
        return dict(
            success=float(all(self.v_found)),
            found_frac=sum(self.v_found) / V,
            golden_frac=golden / V,
            time_to_locate=float(np.mean(found_t)),
            coverage=cov,
            redundancy=(self.stats["revisits"] + self.stats["collisions"]) / moves,
            crashes=self.stats["crashes"],
            downs=self.stats["downs"],
            steps=self.t,
        )

    def snapshot(self):
        return dict(
            t=self.t, pos=list(self.pos), battery=list(self.battery), alive=list(self.alive),
            assign=list(self.assign), comm=list(self.comm), wind=self.wind,
            covered=self.covered.copy(), found=list(self.v_found), relay=self.relay_pos,
        )
