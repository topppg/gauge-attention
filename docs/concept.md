# The idea

Gauge attention treats position encoding as transport between local coordinate
frames. This repository implements the simplest real-valued version of that
idea: every pair of hidden coordinates is represented as a 2D plane, so its
transport is an ordinary rotation.

## A connection on sequence edges

Let `A[h, p, k]` be a learned angle on the edge from sequence position `k` to
`k + 1`, for attention head `h` and coordinate pair `p`. The accumulated phase
at position `i` is

```text
theta[h, p, 0] = 0
theta[h, p, i] = sum(A[h, p, k], k = 0 ... i - 1)
```

The transport phase from a query at `i` to a key at `j` is

```text
Phi[h, p, i, j] = theta[h, p, j] - theta[h, p, i].
```

This immediately gives two useful identities:

```text
Phi(i, j) = -Phi(j, i)
Phi(i, j) + Phi(j, k) = Phi(i, k).
```

The code tests both identities directly.

## Relation to RoPE

When every edge has the same value `omega[p]`, the accumulated phase is
`theta[p, i] = i * omega[p]`. Choosing the standard RoPE frequencies for
`omega` recovers RoPE exactly. A learned, nonconstant connection is more
general, but it is not purely a function of relative distance: intervals of
the same length can cross different learned edges.

Queries and keys are rotated by their absolute phases. Their dot product then
contains the transport between their positions without materializing the full
pairwise phase tensor during the forward pass.

## Values and boundaries

By default only queries and keys are rotated, matching normal RoPE behavior.
`rotate_values=True` also transports values and then brings each result back to
the query frame. It is optional and off by default.

The learned connection has a finite number of edges. The default boundary mode
raises an error beyond that range. Two explicit alternatives are available:

- `repeat_last`: continue using the last learned edge value.
- `zero`: append zero increments, so the accumulated phase stays fixed.

These are boundary definitions, not claims about behavior on longer sequences.

## Scope

This is a research prototype and reference implementation. It provides the
mechanism and exact mathematical checks, but no training recipe, benchmark, or
evidence that the parameterization improves a model. It uses independent 2D
rotations, not a non-Abelian gauge group, and its attention calculation is kept
explicit for readability rather than optimized for production.
