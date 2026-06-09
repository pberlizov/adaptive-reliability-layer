# Fashion-MNIST Scale-Up

## Purpose

This benchmark is the first scale-up from the earlier tabular and toy-image prototypes to a more realistic public dataset and a genuinely convolutional model with batch normalization.

The goal is not just to get another benchmark number. It is to test whether the existing controller architecture:

- still functions against a real image model
- still benefits from BN refresh and bounded adapter updates
- still reduces runaway sequential risk under substantial shift
- still outperforms naive continual adaptation

## Design

### Dataset

We use a fine-grained binary subset of `FashionMNIST`:

- negative classes: `0` and `2`
- positive classes: `4` and `6`

This makes the task materially harder than a broad “footwear vs everything else” split and avoids an artificially separable problem.

### Model

The benchmark now supports two image backbones:

- `convnet`: the fast research loop
- `resnet_small`: the stronger confirmation backbone

The default source model is a small convolutional network with:

- convolutional blocks with batch normalization
- a learned projector
- an adapter block reserved for test-time updates
- a binary classification head

Only the adapter and head are updated at test time.

### Stream regimes

The streaming benchmark includes:

- `stable`
- `brightness_noise`
- `label_shift`
- `inverted_occlusion`
- `brightness_recurrence`
- `translated_blur`

These are simple synthetic shifts, but applied to a real public dataset and a nontrivial model.

The benchmark also now supports two severity profiles:

- `standard`
- `harsh`

The default saved confirmation suite now focuses on the fast path:

- `convnet`
- representative controller subset
- `standard` plus `harsh`
- two seeds

## Representative Result

Most recent `standard` convnet run:

- `frozen`: `0.736`
- `naive`: `0.715`
- `controller`: `0.736`
- `multi_action`: `0.736`
- `bandit`: `0.737`
- `specialist_memory`: `0.734`
- `hybrid`: `0.735`

Operationally:

- `frozen` mean risk capital: `74.592`
- `controller` mean risk capital: `7.111`
- `bandit` mean risk capital: `4.291`

Most recent `harsh` convnet run:

- `frozen`: `0.700`
- `naive`: `0.646`
- `multi_action`: `0.705`
- `specialist_memory`: `0.707`

Operationally:

- `frozen` mean risk capital: `74.592`
- `bandit` mean risk capital: `3.901`
- `multi_action` mean risk capital: `8.486`

## Interpretation

This benchmark says three useful things.

1. The architecture is portable.  
   The controller layer is no longer tied to tabular MLPs.

2. Naive continual adaptation remains brittle.  
   The always-adapt baseline is still worse than the safer controllers.

3. The main value proposition remains safety-governed adaptation rather than raw accuracy jumps.  
   The controller family roughly matches frozen accuracy while cutting sequential risk by an order of magnitude.

4. The harsher profile is useful.  
   It creates a clearer predictive gap between frozen and controller-guided behavior than the standard profile.

5. The reduced image suite is practical.  
   We now have a saved confirmation artifact that is heavy but still reasonable enough to run, without paying the full cost of the slower `resnet_small` track every time.

## Next Step

The next logical upgrade is now:

- a reduced but repeatable image scale-up suite
- or a move from this custom CNN to confirmation runs on the `resnet_small` path

The important point is that we now have a real-image benchmark that justifies that next step.
