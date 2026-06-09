# Prototype Plan

## Prototype Objective

Build a minimal end-to-end research prototype that demonstrates the control loop:

1. observe streaming data
2. detect shift
3. decide whether to adapt
4. adapt conservatively
5. measure whether the adaptation helped or hurt
6. expose prediction plus reliability metadata

## V1 System Components

### 1. Stream Environment

A synthetic environment that emits batches across multiple regimes:

- stable source distribution
- gradual covariate drift
- abrupt shift
- recurring regime

This keeps the first prototype simple while still exercising continual adaptation behavior.

### 2. Base Predictor

A lightweight classifier with:

- an explicit parameter vector
- a `predict_proba` interface
- an `adapt` interface for bounded updates

For the first pass, a linear or logistic model is enough.

### 3. Shift Monitor

A streaming monitor that tracks:

- batch mean drift
- batch variance drift
- simple latent or feature-distance drift

The monitor outputs:

- a scalar shift score
- a boolean alert
- monitor metadata for analysis

### 4. Adaptation Controller

A policy that chooses among:

- no action
- confidence recalibration
- bounded parameter update
- reset to source parameters

The policy is gated by drift severity and simple safety rules.

### 5. Safety Layer

The first version includes:

- source parameter snapshot
- bounded step size
- adaptation cooldown
- rollback/reset trigger when drift remains high after repeated updates

### 6. Uncertainty Wrapper

A lightweight uncertainty signal composed from:

- model confidence
- drift score
- adaptation state

This is not yet full conformal inference, but it provides a surface we can later replace with stronger methods.

### 7. Evaluation Harness

We need to compare:

- frozen model
- naive adaptation
- safety-gated adaptation

Metrics:

- cumulative accuracy
- regime-wise accuracy
- number of alerts
- number of adaptations
- harmful adaptation count
- recovery time after shift

## Immediate Build Milestones

### Milestone 1

- create package scaffold
- implement synthetic stream generator
- implement simple classifier
- implement monitor
- implement controller
- implement simulation script

### Milestone 2

- add benchmark metrics and richer logging
- add replay anchors or source-buffer proxy
- compare frozen vs adaptive baselines

### Milestone 3

- add latent-space monitoring
- add a better uncertainty wrapper
- add one graph-structured toy benchmark

## Research Direction After V1

Once the simple loop works, we should choose one of two deeper directions:

1. safety and control:
focus on the controller, rollback logic, resets, and bounded optimization

2. structured shift:
focus on graph-aware monitoring and adaptation in relational domains

My current recommendation is to start with the safety/control direction because it is the cleanest systems contribution and transfers across domains.
