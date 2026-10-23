# AeroMind: multi-agent RL for high-altitude search & rescue

Tabular RL written from scratch (numpy only). No RL library, no neural network.

| Agent | Count | Algorithm |
|---|---|---|
| UAV searcher | 4 | Independent Q-Learning (own Q-table each) |
| Coordinator | 1 | Expected SARSA (assigns UAVs to 3x3 zones every 4 min) |
| Relay UAV | 1 (optional) | Q-Learning (restores radio links in comm-loss zones) |

## Files
- `env.py`       simulator: terrain, wind, battery, comm-loss zones, victims, rewards
- `agents.py`    Q-Learning / Expected-SARSA agents, random + greedy baselines, episode runner
- `train.py`     training script (also used by the app)
- `evaluate.py`  compare teams on identical seeds, comm-loss sweep
- `app.py`       Streamlit app (replay, live training, evaluation, design tab)
- `requirements.txt`

Keep all files in one folder (create empty `models/` and `logs/` folders, or let train.py create them).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
In the app open **Train agents**, pick a configuration, press **Train from scratch** (default 1,500
episodes), then use **Mission replay** and **Evaluate**.

## Train from the command line (all settings configurable)
```bash
python train.py --config coord --episodes 1500 --alpha 0.15 --gamma 0.9 --seed 0
python train.py                       # all three configurations, 1,500 episodes each
```
`--config` is `iql` (searchers only), `coord` (+ coordinator) or `full` (+ relay UAV).
Models are saved to `models/<config>.pkl`, logs to `logs/train_<config>.csv`; the app loads them
automatically. More episodes (e.g. 10000) give stronger policies.

## Evaluate
```bash
python evaluate.py        # random, greedy and every trained model found in models/
```
Edit `Config` in `env.py` (grid size, victims, battery, wind, comm-loss zones) for other settings.

## Deploy on Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo (include `models/*.pkl` if you want pre-trained teams).
2. share.streamlit.io -> New app -> choose repo, branch, main file `app.py` -> Deploy.
Without model files the app still works: train in the Train tab.

## Notes
- Searcher count is fixed at 4 in the app because each searcher owns a Q-table; change `Config.n_search` and retrain for another number.
- Whether the coordinator or relay beats plain IQL depends on training length and seed; compare them in the Evaluate tab.
