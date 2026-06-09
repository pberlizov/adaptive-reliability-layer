# Graph Shift Track

## Purpose

This track is the first graph-native extension of the adaptive reliability layer.

The goal is not yet to prove that the current controller family can fully recover accuracy under structural graph drift. The nearer-term goal is to test whether the same architecture can:

- detect graph-structural shift explicitly
- distinguish structural shift from ordinary feature drift
- reduce unsafe behavior when the deployed model is outside its structural competence zone

## Current Design

The current benchmark:

- generates dynamic synthetic graphs with node features and binary node labels
- trains a source model on node features
- monitors both:
  - feature/output drift
  - topology drift

Topology monitoring currently uses:

- edge density
- mean degree
- degree spread
- clustering
- spectral gap

The same controller family is then reused:

- `frozen`
- `naive`
- `controller`
- `multi_action`
- `bandit`
- `specialist_memory`
- `hybrid`

## Hardening Update

The first version was too easy: frozen performance remained near-perfect even under topology shift.

We hardened it by:

- reducing source class-margin strength
- increasing feature noise
- introducing community and degree confounding
- making topology regimes actively flip or corrupt learned structural correlations
- shrinking the labeled source pool

This made the benchmark materially more informative.

## Representative Result

Recent 48-step run with 48-node graphs:

- `frozen`: overall `0.824`
- `topology_rewire` regime under `frozen`: `0.604`
- `bandit`: overall `0.820`
- `controller`: overall `0.811`

Operationally:

- `frozen` mean risk capital: `18.166`
- `controller` mean risk capital: `3.513`
- `bandit` mean risk capital: `4.358`

## Interpretation

This track now says something useful:

- structural rewiring genuinely degrades the frozen model
- the controller family still behaves more like a safety layer than a performance recovery engine
- graph-aware monitoring is now justified, because the benchmark has real topology-induced failure

That is a good intermediate outcome. It means the graph track is no longer trivial, even if it is not yet the strongest adaptation benchmark in the repo.
