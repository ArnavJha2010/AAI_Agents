"""
AeroMind - multi-agent reinforcement learning for high-altitude search & rescue.
Streamlit front-end:  replay a mission, train the agents live, evaluate, read the design.
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

import train as T
import evaluate as E
from env import (SearchEnv, Config, SEARCH_ACTION_NAMES, RELAY_ACTION_NAMES, WIND_NAMES,
                 N_ZONES, ZONES_PER_SIDE, COORD_HOLD, COORD_PERIOD)
from agents import run_episode

st.set_page_config(page_title="AeroMind", page_icon="🛰️", layout="wide")

METHODS = {
    "random": "Random search (baseline)",
    "greedy": "Greedy heuristic (baseline, no learning)",
    "iql": "IQL searchers",
    "coord": "IQL searchers + Expected-SARSA coordinator",
    "full": "IQL searchers + coordinator + relay UAV",
}
SESSION_KEYS = ("models", "logs", "frames", "static", "eval_df", "sweep_df")
for k in SESSION_KEYS:
    st.session_state.setdefault(k, {} if k in ("models", "logs") else None)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner=False)
def shipped_model(key):
    return T.load(key)


def get_team(key, cfg):
    if key in ("random", "greedy"):
        return E.get_team(key, cfg)
    if key in st.session_state["models"]:
        team = st.session_state["models"][key]
        team.cfg = cfg
        return team
    team = shipped_model(key)
    if team is not None:
        team.cfg = cfg
    return team


def have_model(key):
    return key in ("random", "greedy") or key in st.session_state["models"] or shipped_model(key) is not None


def draw(static, fr, selected=None):
    n = static["n"]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    img = np.ones((n, n, 3)) * np.array([0.94, 0.96, 0.98])
    img[fr["covered"]] = [0.75, 0.88, 0.96]
    img[static["nofly"]] = [0.2, 0.23, 0.29]
    ax.imshow(img, extent=(-0.5, n - 0.5, n - 0.5, -0.5), interpolation="nearest")
    lost = np.where(static["comm_lost"], 1.0, np.nan)
    ax.imshow(lost, cmap=ListedColormap(["#e4572e"]), alpha=0.22,
              extent=(-0.5, n - 0.5, n - 0.5, -0.5), interpolation="nearest")
    side = n // ZONES_PER_SIDE
    for k in range(1, ZONES_PER_SIDE):
        ax.axhline(k * side - 0.5, color="#5b6b7a", lw=0.8, ls=":")
        ax.axvline(k * side - 0.5, color="#5b6b7a", lw=0.8, ls=":")
    ax.text(-0.35, -0.2, "BASE", fontsize=7, color="#0b3d63", weight="bold", va="top")

    for v, (vr, vc) in enumerate(static["victims"]):
        if fr["found"][v]:
            ax.scatter(vc, vr, marker="*", s=260, color="#2a9d4a", edgecolor="white", zorder=5)
        else:
            ax.scatter(vc, vr, marker="v", s=90, color="#d62828", alpha=0.55, zorder=4)

    if static["use_relay"]:
        rr, rc = fr["relay"]
        rad = static["relay_radius"]
        ax.add_patch(plt.Rectangle((rc - rad - 0.5, rr - rad - 0.5), 2 * rad + 1, 2 * rad + 1,
                                   fill=False, ec="#7b2cbf", lw=1.2, ls="--"))
        ax.scatter(rc, rr, marker="D", s=120, color="#7b2cbf", edgecolor="white", zorder=6)

    cols = ["#f77f00", "#0077b6", "#43aa8b", "#c9184a", "#6a4c93", "#adb5bd"]
    for i, (r, c) in enumerate(fr["pos"]):
        z = fr["assign"][i]
        if z >= 0 and fr["alive"][i]:
            zr, zc = static["zone_centres"][z]
            ax.plot([c, zc], [r, zr], color=cols[i % 6], lw=0.8, alpha=0.35, ls="--")
        if not fr["alive"][i]:
            ax.scatter(c, r, marker="x", s=90, color="black", zorder=7)
            continue
        ring = "#e4572e" if not fr["comm"][i] else "white"
        size = 220 if i == selected else 150
        ax.scatter(c, r, s=size, color=cols[i % 6], edgecolor=ring, linewidth=2.2, zorder=8)
        ax.text(c, r, str(i + 1), color="white", ha="center", va="center", fontsize=8, weight="bold", zorder=9)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"t = {fr['t']} min   |   wind: {WIND_NAMES[fr['wind']]}   |   "
                 f"found {sum(fr['found'])}/{len(fr['found'])}", fontsize=10)
    fig.tight_layout()
    return fig


def static_from_env(env):
    return dict(n=env.n, nofly=env.nofly.copy(), comm_lost=env.comm_lost.copy(), victims=list(env.victims),
                zone_centres=list(env.zone_centres), use_relay=env.cfg.use_relay,
                relay_radius=env.cfg.relay_radius, found_t=list(env.v_found_t),
                n_comm_zones=env.n_comm_zones, golden=env.cfg.golden)


def rolling(df, col, w=100):
    return df[col].rolling(w, min_periods=1).mean()


# -------------------------------------------------------------------- header
st.title("🛰️ AeroMind")
st.caption("Multi-agent reinforcement learning for high-altitude search & rescue · "
           "UAV searchers (Independent Q-Learning) · coordinator (Expected SARSA) · relay UAV (Q-Learning) · "
           "all tabular RL written from scratch")

tab_replay, tab_train, tab_eval, tab_design = st.tabs(
    ["🎬 Mission replay", "🧠 Train agents", "📊 Evaluate", "📐 Design (MDP & Markov game)"])

# ====================================================================== REPLAY
with tab_replay:
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
    method = c1.selectbox("Team", list(METHODS), index=3, format_func=METHODS.get, key="rp_method")
    seed = c2.number_input("Scenario seed", 0, 10**6, 7, key="rp_seed")
    zones = c3.selectbox("Comm-loss zones", [-1, 0, 1, 2, 3], index=3,
                         format_func=lambda z: "random" if z < 0 else str(z), key="rp_zones")
    run = c4.button("▶ Run mission", type="primary")

    if run:
        if not have_model(method):
            st.error("No trained model for this team yet. Train it in the 'Train agents' tab (or run train.py).")
        else:
            cfg = T.make_cfg(method if method in T.CONFIGS else "iql", comm_zones=zones)
            cfg.use_relay = method == "full"
            team = get_team(method, cfg)
            env = SearchEnv(cfg)
            m, frames = run_episode(env, team, seed=int(seed), learn=False, record=True)
            st.session_state["frames"] = dict(frames=frames, metrics=m, method=method)
            st.session_state["static"] = static_from_env(env)
            st.session_state["rp_step"] = 0

    data, static = st.session_state["frames"], st.session_state["static"]
    if data is None:
        st.info("Pick a team and press **Run mission**. Red triangles are buried victims, green stars are located "
                "victims, the red-tinted area has no radio coverage, dark cells are no-fly cliffs, dotted lines "
                "show the coordinator's zone assignments.")
    else:
        frames, m = data["frames"], data["metrics"]
        mc = st.columns(6)
        mc[0].metric("Victims located", f"{sum(frames[-1]['found'])}/{len(frames[-1]['found'])}")
        mc[1].metric("Mission time", f"{m['steps']} min")
        mc[2].metric("In golden window", f"{m['golden_frac'] * 100:.0f}%")
        mc[3].metric("Area covered", f"{m['coverage'] * 100:.0f}%")
        mc[4].metric("Redundancy", f"{m['redundancy']:.2f}")
        mc[5].metric("UAVs lost", m["downs"])

        left, right = st.columns([1.15, 1])
        ctrl = left.columns([3, 1, 1])
        step = ctrl[0].slider("Mission minute", 0, len(frames) - 1, key="rp_step")
        play = ctrl[1].button("⏵ Play")
        sel = ctrl[2].selectbox("Inspect UAV", [1, 2, 3, 4], key="rp_uav") - 1
        holder = left.empty()

        def show(k):
            fig = draw(static, frames[k], selected=sel)
            holder.pyplot(fig, clear_figure=True)
            plt.close(fig)

        show(step)
        fr = frames[step]
        with right:
            st.markdown("**Fleet status (after this minute)**")
            rows = []
            for i in range(len(fr["pos"])):
                rows.append({
                    "UAV": i + 1,
                    "status": "down" if not fr["alive"][i] else ("link lost" if not fr["comm"][i] else "linked"),
                    "battery": fr["battery"][i],
                    "action": SEARCH_ACTION_NAMES[fr["acts"][i]],
                    "reward": round(fr["rew"][i], 2),
                    "zone": "-" if fr["assign"][i] < 0 else str(fr["assign"][i] + 1),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            qs = fr["info"].get("q") if fr.get("info") else None
            if qs and qs[sel] is not None:
                st.markdown(f"**How UAV {sel + 1} decided** (learned Q-values for its current state)")
                qdf = pd.DataFrame({"Q-value": qs[sel]}, index=SEARCH_ACTION_NAMES)
                st.bar_chart(qdf, height=210)
                st.caption(f"Chosen action: **{SEARCH_ACTION_NAMES[fr['acts'][sel]]}** (highest Q-value = greedy policy).")
            elif data["method"] in ("random", "greedy"):
                st.caption("Baseline team: no Q-values to show (no learning).")

            ev = []
            for k, f in enumerate(frames[: step + 1]):
                co = (f.get("info") or {}).get("coord")
                if co and co["searcher"] is not None:
                    ev.append(f"t={f['t'] - 1}: coordinator sent UAV {co['searcher'] + 1} to zone {co['action'] + 1}")
            if ev:
                st.markdown("**Coordinator log**")
                st.text("\n".join(ev[-8:]))

        if play:
            for k in range(step, len(frames)):
                show(k)
                time.sleep(0.12)

# ======================================================================= TRAIN
with tab_train:
    st.markdown(
        "Train the agents **live** with tabular RL. Agents start with empty Q-tables and an exploration rate "
        "of 1.0 (pure trial and error), then learn from rewards while ε decays. "
        "Every episode is a new random mountain: terrain, victims, wind and comm-loss zones all change.")
    a, b, c, d, e = st.columns(5)
    cfg_key = a.selectbox("Configuration", list(T.CONFIGS), index=1, format_func=lambda k: METHODS[k], key="tr_cfg")
    episodes = b.slider("Episodes", 200, 6000, 1500, 100)
    alpha = c.slider("Learning rate α", 0.02, 0.5, 0.15, 0.01)
    gamma = d.slider("Discount γ", 0.5, 0.99, 0.9, 0.01)
    tseed = e.number_input("Seed", 0, 1000, 0, key="tr_seed")
    st.caption("Roughly 25 ms per episode on a laptop. Short runs learn the basics; for stronger policies use "
               "`python train.py --episodes 10000` and place the resulting models/*.pkl next to app.py.")

    if st.button("🚀 Train from scratch", type="primary"):
        bar = st.progress(0.0, text="Training...")
        chart = st.empty()
        info = st.empty()
        buf = []

        def cb(ep, tot, row):
            buf.append(row)
            if ep % 50 == 0 or ep == tot:
                bar.progress(ep / tot, text=f"Episode {ep}/{tot}   ε = {row['epsilon']:.2f}")
                df = pd.DataFrame(buf)
                chart.line_chart(pd.DataFrame({
                    "success rate (rolling 100)": rolling(df, "success"),
                    "victims in golden window (rolling 100)": rolling(df, "golden_frac"),
                }), height=240)
                info.caption(f"Q-table entries across searchers: {int(row['q_states']):,}")

        team, log = T.train(cfg_key, episodes, alpha, gamma, seed=int(tseed), progress=cb)
        st.session_state["models"][cfg_key] = team
        st.session_state["logs"][cfg_key] = log
        st.success(f"Training finished. This trained model is now used in the Replay and Evaluate tabs "
                   f"for '{METHODS[cfg_key]}'.")

    st.markdown("#### Learning curves")
    logs = {}
    for k in T.CONFIGS:
        if k in st.session_state["logs"]:
            logs[f"{k} (your run)"] = st.session_state["logs"][k]
        else:
            p = os.path.join(T.LOG_DIR, f"train_{k}.csv")
            if os.path.exists(p):
                logs[f"{k} (train.py log)"] = pd.read_csv(p)
    if logs:
        metric = st.selectbox("Metric", ["success", "golden_frac", "time_to_locate", "return", "redundancy", "epsilon"],
                              format_func=lambda m: {"success": "Success rate (all victims located)",
                                                     "golden_frac": "Victims located within the golden window",
                                                     "time_to_locate": "Average time-to-locate (min)",
                                                     "return": "Searcher return (sum of rewards)",
                                                     "redundancy": "Redundancy (revisits + overlaps per move)",
                                                     "epsilon": "Exploration rate ε"}[m])
        curve = pd.DataFrame({name: rolling(df, metric, 200).reset_index(drop=True) for name, df in logs.items()})
        st.line_chart(curve, height=300)
        st.caption("Rolling mean over 200 episodes (exploration is still on during training, so evaluation numbers are better).")

# ====================================================================== EVALUATE
with tab_eval:
    st.markdown("Compare teams on **identical random scenarios** (same seeds, ε = 0, learning off).")
    a, b, c = st.columns([3, 1, 1])
    avail = [k for k in METHODS if have_model(k)]
    picked = a.multiselect("Teams", avail, default=avail, format_func=METHODS.get)
    n_ep = b.slider("Episodes", 20, 300, 100, 10)
    ez = c.selectbox("Comm-loss zones", [-1, 0, 1, 2, 3], format_func=lambda z: "random" if z < 0 else str(z), key="ev_z")
    cc1, cc2 = st.columns(2)
    if cc1.button("Run comparison", type="primary") and picked:
        with st.spinner("Running episodes..."):
            st.session_state["eval_df"] = E.evaluate(picked, n_ep, comm_zones=ez, models=st.session_state["models"])
    if cc2.button("Robustness sweep (0-3 comm-loss zones)") and picked:
        with st.spinner("Sweeping comm-loss zones..."):
            st.session_state["sweep_df"] = E.comm_sweep(picked, max(20, n_ep // 2), models=st.session_state["models"])

    df = st.session_state["eval_df"]
    if df is not None:
        show = df.drop(columns=["key", "episodes"]).rename(columns={
            "method": "Team", "success": "Success rate", "found_frac": "Victims found",
            "golden_frac": "In golden window", "time_to_locate": "Time-to-locate (min)",
            "coverage": "Area covered", "redundancy": "Redundancy", "crashes": "Crashes / mission",
            "downs": "UAVs lost / mission", "return": "Searcher return"})
        st.dataframe(show.round(3), hide_index=True)
        g1, g2, g3 = st.columns(3)
        g1.markdown("**Time-to-locate (min, lower is better)**")
        g1.bar_chart(df.set_index("method")["time_to_locate"], height=240)
        g2.markdown("**Victims located in golden window**")
        g2.bar_chart(df.set_index("method")["golden_frac"], height=240)
        g3.markdown("**Redundancy (lower is better)**")
        g3.bar_chart(df.set_index("method")["redundancy"], height=240)

    sw = st.session_state["sweep_df"]
    if sw is not None:
        st.markdown("#### Robustness to communication loss")
        s1, s2 = st.columns(2)
        s1.markdown("**Time-to-locate vs. comm-loss zones**")
        s1.line_chart(sw.pivot(index="comm_zones", columns="method", values="time_to_locate"), height=260)
        s2.markdown("**Golden-window fraction vs. comm-loss zones**")
        s2.line_chart(sw.pivot(index="comm_zones", columns="method", values="golden_frac"), height=260)

# ======================================================================= DESIGN
with tab_design:
    st.subheader("Agents")
    st.dataframe(pd.DataFrame([
        ["UAV searcher", "x4", "Locate victims fast, avoid cliffs, return before the battery dies",
         "Hint to nearest signal, direction to nearest unexplored cell, battery level, link up?, wind, 8-cell terrain window",
         "8 moves, scan, return-to-base", "Independent Q-Learning (own Q-table each)"],
        ["Coordinator", "x1", "Minimise expected time-to-locate for the whole fleet",
         "Least-covered zone, coverage spread, latest signal ping zone, number of linked UAVs",
         "Send nearest linked UAV to zone 1-9, or hold (every 4 min)", "Expected SARSA"],
        ["Relay UAV", "x1 (optional)", "Keep searchers inside radio coverage",
         "Direction and distance to the nearest UAV without a link, own zone",
         "8 moves, hold", "Q-Learning"],
    ], columns=["Agent", "Count", "Goal", "Observes", "Acts", "Learns via"]), hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Environment")
        st.markdown(
            f"- **Grid:** 15 x 15 cells, 1 step = 1 minute, base at the top-left corner (recharges UAVs)\n"
            "- **Hazards:** ~7% no-fly cells (cliffs / crevasses), random each episode\n"
            "- **Wind:** calm / N / E / S / W, changes randomly, pushes a UAV one cell with probability 0.3\n"
            "- **Comm-loss zones:** 0-3 random rectangles with no radio link (relay UAV can restore it)\n"
            "- **Victims:** 3 buried at t=0. Survival probability = exp(-(t/60)²). Thermal signal = exp(-t/120)\n"
            "- **Battery:** 70 minutes of flight; a UAV that runs dry is lost\n"
            "- **Termination:** all victims located, or 200 minutes, or every UAV lost")
        st.subheader("Rewards")
        st.dataframe(pd.DataFrame([
            ["New cell covered", "+1.0 (+0.5 outside assigned zone)"],
            ["Revisit a covered cell", "-0.3"],
            ["Victim located", "+60 x survival probability"],
            ["Energy per minute / scan", "-0.1 / -0.2 extra"],
            ["Crash into cliff / edge", "-3"],
            ["Two UAVs in one cell", "-2 each"],
            ["Battery empty", "-20"],
            ["Recharge on low battery", "+2"],
            ["Progress home when battery critical", "+0.5 per cell closer"],
            ["Coordinator (shared)", "detections + 0.1 x new cells - 0.05 x urgency of unfound victims"],
            ["Relay UAV", "fraction of searchers with a link - 0.02 per move"],
        ], columns=["Event", "Reward"]), hide_index=True)
    with c2:
        st.subheader("RL formulation (single searcher MDP)")
        st.markdown(
            "- **S:** (thermal hint, goal direction, battery level, link, wind, terrain window)\n"
            "- **A:** 8 moves, scan, return-to-base\n"
            "- **R:** coverage + urgency-weighted detection − energy − revisits − crashes\n"
            "- **P:** motion perturbed by wind drift; victim signal decays\n"
            "- **γ:** 0.9  ·  **π:** ε-greedy over Q during learning, greedy afterwards")
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\,]")
        st.markdown("Coordinator update (Expected SARSA):")
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,R+\gamma\sum_{a'}\pi(a'|s')Q(s',a')-Q(s,a)\,]")
        st.subheader("Markov game")
        st.markdown(
            "- **N** = 4 searchers + 1 coordinator (+ 1 relay)\n"
            "- Fully **cooperative** objective, decentralised execution\n"
            "- Searchers get **individual** shaping rewards, coordinator and relay get **team** rewards\n"
            "- Communication is **required** for coordination, so it is lost inside comm-loss zones. "
            "A UAV without a link keeps its last zone and its own coverage map and carries on alone.")
        st.graphviz_chart("""
        digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f1f8", fontsize=11];
          Env [label="ENVIRONMENT\\n(terrain, wind, victims, comm zones)", fillcolor="#cfe3f1"];
          C [label="Coordinator\\nExpected SARSA"]; R [label="Relay UAV\\nQ-Learning"];
          S1 [label="Searcher 1\\nIQL"]; S2 [label="Searcher 2\\nIQL"]; S3 [label="Searcher 3\\nIQL"]; S4 [label="Searcher 4\\nIQL"];
          Env -> S1 [label="obs"]; Env -> S2; Env -> S3; Env -> S4;
          S1 -> Env [label="action"]; S2 -> Env; S3 -> Env; S4 -> Env;
          Env -> C [label="team state"]; C -> Env [label="zone assignment"];
          Env -> R [label="link state"]; R -> Env [label="move relay"]; }
        """)
    st.caption("Everything is learned with tabular Q-Learning / Expected SARSA implemented from scratch with numpy. "
               "No RL library and no pre-trained neural network is used. Model files (models/*.pkl) are simply the "
               "Q-tables produced by train.py or the Train tab.")
