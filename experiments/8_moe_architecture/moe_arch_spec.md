# moe_arch_spec.md
# Mixture-of-Experts Architecture Specification

**Owner:** MoE Architecture Team  

---

## 1. Objective

The objective of this specification is to define and freeze a **compute-efficient, routing-safe, FFN-only Mixture-of-Experts (MoE) architecture** that:

- Fits within hard FLOPs/token, memory, and throughput ceilings
- Works cleanly with the approved growth cadence:
  
  **1B Dense → 3B MoE-small → 8B Dense-deep → 70B MoE-large**
- Uses **MoE in every transformer layer**
- Introduces **null experts inside top-k routing** (native competition)
- Uses **loss-free routing control** (bias-only; no auxiliary losses)
- Scales capacity via **expert explosion**, not increased per-token compute
- Passes all routing health gates defined by Team 7

This spec is designed to prevent routing collapse, expert starvation, and late-stage irreversibility.

---

## 2. Growth Cadence

| Stage | Model Type | Purpose | Architectural Rule |
|------|------------|---------|--------------------|
| 1B | Dense | Representation formation | No MoE |
| 3B | MoE-small | Routing & expert identity learning | First MoE introduction |
| 8B | Dense-deep | Consolidation | **MoE topology frozen** |
| 70B | MoE-large | Capacity scaling | **Only expert count scales** |

### Binding Conditions
- MoE is **FFN-only**
- MoE is present in **every layer**
- Expert configuration (counts, k, rho) is **frozen after 3B**
- 8B stage **must not change expert topology**
- 70B stage **must not introduce new routing mechanisms**

Failure of routing health gates at any stage blocks progression.

---

## 3. Transformer Block Definition

Each transformer layer follows the structure:

[Attention]
↓
[MoE-FFN]
↓
[Residual Add]
↓
[Normalization]


### MoE-FFN Composition

Each MoE-FFN layer contains:

1. **Shared Experts**
   - Always active
   - Provide global, non-specialized capacity
2. **Routed Experts**
   - Selected via router top-k
3. **Null Experts**
   - Zero-compute
   - Compete natively inside top-k

No layer uses dense FFNs once MoE is enabled.

---

## 4. Router Architecture (Loss-Free)

### 4.1 Logit Computation

For each token representation `x`:

logits_i = W_router(x) + bias_i


Where:
- `W_router` is a learned linear projection
- `bias_i` is a **mutable, external control surface**

### 4.2 Selection

- Router computes logits over **N + M** entries:
  - `N` real routed experts
  - `M` null expert duplicates
- Router selects **top-k_max**
- Selected null experts produce **zero output**
- Routing weights are **renormalized over real experts only**

### 4.3 Explicitly Disallowed

The following are **out of spec**:

- Auxiliary router losses (load balancing loss, z-loss, entropy loss)
- Token dropping
- External null paths
- Two-stage routing
- Expert-dependent loss terms

All routing control must be achieved **only via bias adjustment**.

---

## 5. Null Expert Semantics (Binding)

Null experts are first-class routing entries and must satisfy:

### Required Properties
- Output is **exact zero tensor**
- Participate **inside top-k**
- Do **not** receive gradients
- Do **not** alter output magnitude after renormalization

### Explicitly Disallowed Null Types
- Identity / copy experts
- Constant non-zero experts
- Residual-pass experts
- Learned null embeddings

### Operating Regime
- Effective sparsity parameter: `rho ∈ [0.5, 0.67]`
- Default: `rho = 0.5`

This regime is chosen to avoid routing collapse and expert resolution loss.

---

## 6. Scaling Invariants (Frozen)

The following invariants are enforced across all MoE stages:

| Invariant | Value |
|----------|-------|
| Routing segments (m) | 4 |
| Expected real experts per token | 8 |
| Shared experts | Decays with scale |
| Expert FFN dimensions | Fixed |
| Scaling axis | Expert count only |

Per-token FLOPs **must not increase with scale**.

---

## 7. Mathematical Routing Parameters

Definitions:

- `m`: routing segments
- `N`: real routed experts
- `M`: null expert duplicates
- `k_max`: top-k slots
- `rho`: expected fraction of real experts
- `E[K_real]`: expected real experts per token

Formulas:

E[K_real] = k_max * rho
M = N * (1 - rho) / rho


With `rho = 0.5` and `E[K_real] = 8`:

k_max = 16
M = N


---

## 8. Stage-Wise Configurations

### 8.1 3B MoE-Small (Routing Learning Stage)

| Parameter | Value |
|---------|-------|
| Routing segments (m) | 4 |
| Real experts (N) | 32 |
| Null experts (M) | 32 |
| Shared experts | 2 |
| k_max | 16 |
| rho | 0.5 |
| E[K_real] | 8 |
| Total active experts | ≈10 |

**Purpose:**  
Learn router geometry and expert identity under tight capacity.

---

### 8.2 8B Dense-Deep (Consolidation Stage)

**Topology identical to 3B MoE-small**

| Parameter | Value |
|---------|-------|
| Real experts (N) | 32 |
| Null experts (M) | 32 |
| Shared experts | 2 |
| k_max | 16 |
| rho | 0.5 |

No expert count, k, or routing changes are permitted.

---

### 8.3 70B MoE-Large (Expert Explosion)

| Parameter | Value |
|---------|-------|
| Routing segments (m) | 4 |
| Real experts (N) | 256–512 |
| Null experts (M) | N |
| Shared experts | 1 |
| k_max | 16 |
| rho | 0.5 |
| E[K_real] | 8 |
| Total active experts | ≈9 |

**Rule:**  
Only `N` scales. All other parameters are frozen.

---

## 9. Routing Health Gates (Binding)

The following metrics must remain within bounds:

- Router entropy does not collapse
- No expert starvation (>5% sustained)
- No polarization (tokens with zero real experts)
- Load imbalance < defined threshold

Failure blocks progression.

---

## 10. Null Routing Targets

Measured by Team 7 telemetry.

| Token Group | Target Null Rate |
|------------|-----------------|
| Junk tokens | 60–80% |
| Natural language | 30–50% |
| Code | <20% |
| Reasoning / math | <15% |

Token groups are defined by Team 6 tokenizer ID bands.

---

## 11. Explicitly Disallowed Features

The following are **NO-GO** in MoE stages:

- Multi-Token Prediction (MTP)
- Auxiliary routing losses
- Dynamic expert dimension scaling
- Token dropping
- Identity / residual nulls

Any inclusion requires architecture review.

---

## 12. Fallback Configurations (Plan B)

Triggered only if routing health gates fail.

### Allowed Adjustments
- Reduce `k_max` to 12
- Increase shared experts by +1
- Reduce null ratio to `M = 0.75N`

Fallback use must be logged and approved.

---

## 13. Validation Requirements

Before advancing stages:

- Training loss matches dense SLM baseline
- Routing health gates pass
- Null absorbs junk at target rates
- Signal token groups are protected
**Dense warmup of at least 20k steps before enabling MoE and null routing in 3B stage, per Null Experts paper recommendations.**
No validation → no scale-up.
