# HN post draft (edit before posting)

Numbers from current run — reproduce with `arl-hn-launch`.

## Title options

Pick one:

1. **Show HN: Reliability layer for fraud ML under delayed labels (open benchmarks, PyPI)**
2. **Fraud models hit 99% accuracy but ops keeps retraining — we published a source-available controller for that**
3. **Adaptive Reliability Layer – controller steering for fraud models when chargeback labels arrive late**

## Post body (markdown for GitHub; paste into HN as plain text)

---

Every time a drift alarm fires at a fraud/AML shop, someone faces a decision: retrain the model, or wait. Retraining means rebuilding the dataset, running backtests, getting model risk sign-off (SR 11-7 if you're regulated), staging a deployment, holding a rollback window. Most drift alarms don't need it — the shift is benign or transient. But the standard tooling (PSI thresholds, ADWIN, scheduled jobs) can't tell the difference, so teams retrain on a schedule and eat the cost.

We built **ARL (Adaptive Reliability Layer)** to solve that. It's a controller that sits beside your deployed fraud model, learns from delayed revealed labels (chargebacks arrive weeks after the transaction), and takes the smallest bounded steering step that stabilizes the model — correction first, explicit mutation only when needed. It defers retrains when they're not needed, recommends them when they are, and logs every decision with rollback metadata.

**The complication:** on public fraud datasets (ULB, IEEE-CIS, PaySim), models score **94–99% accuracy** frozen. Raw accuracy is useless as a signal — everything looks fine until it doesn't. We measure:

- **Retrain deferral** — how many retrain cycles avoided without harming accuracy
- **Proxy risk reduction** — the strongest reduction among martingale capital, alert rate, and retrain recommendations on the monitoring stream
- **Utility** — accuracy minus operational costs: false alarms, unnecessary retrains, resets

**Results on open temporal replay** (delayed labels, 12-step delay, reproducible):

- Beats **scheduled retrain** on utility on **3/3 core** fraud sources (ULB, IEEE-CIS, PaySim)
- **6–9% proxy risk reduction** vs frozen across all 3 core sources (ULB 7.2%, IEEE 8.7%, PaySim 6.0%)
- Detection accuracy flat (expected — that's your model's job, not ARL's)
- Elliptic (Bitcoin blockchain) and BAF also included as extended tier
- On the flagship fraud streams, the win mostly comes from **narrow controller steering** rather than frequent explicit mutate-the-model actions

**Honest limits:** Elliptic's temporal structure is driven by illicit cluster exposure windows rather than the gradual covariate drift ARL is designed for — it's a structurally different problem. Hard-slice fraud detection metrics mostly tie across methods (accuracy saturates). This is an **ops/reliability** story. Public CSV replay, not your production traffic.

**If you're at a fraud/AML shop:** ARL can run in shadow mode against your own replay CSV (`arl-customer-replay`) before you commit to anything.

Install + try in ~2 minutes (no downloads):

```bash
mkdir arl-demo && cd arl-demo
pip install "adaptive-reliability-layer[torch,serving]>=0.3.4"
arl-demo
```

Full five-dataset suite: `arl-hn-launch` (~30–90 min). Results land in `results/hn_launch/comparison_table.md`.

License note: the repo is **source-available under BUSL-1.1**. Demo,
benchmarking, research, and internal evaluation are fine; production use,
managed-service use, and customer-facing deployment require a commercial
license.

- **PyPI:** https://pypi.org/project/adaptive-reliability-layer/
- **Sidecar:** shadow → bounded_auto, `/v1/batch` + delayed `/v1/batch/{step}/labels`

Looking for teams with delayed-label fraud/risk models — happy to compare replay methodology.

---

## First comment (post immediately after submission)

```
Author here. Quick links:

• PyPI: pip install "adaptive-reliability-layer[torch,serving]>=0.3.4"
• Try first: arl-demo  (~2–5 min, PaySim toy, no downloads)
• Full reproduce: arl-hn-launch (~30–90 min)
• Comparison table: results/hn_launch/comparison_table.md

README has sidecar curl examples. AMA on delayed labels / retrain policy / why we stopped optimizing accuracy on these streams.
```

## Show HN tips

- Reply fast to first 10 comments
- Don't argue about accuracy — redirect to utility + delayed labels + vs scheduled retrain
- If asked for customer proof: shadow replay on their CSV (`arl-customer-replay`)
