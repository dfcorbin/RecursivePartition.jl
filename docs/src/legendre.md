# Legendre Polynomials

## Tutorial

The Legendre Polynomials are a set of univariate polynomials defined on the
interval ``[-1, 1]``. Let \
``ϕ_j(x) : [-1, 1] → \mathbb{R}`` denote
the order ``j \in \mathbb{N}_{≧0}`` Legendre Polynomial. The Legendre Polynomials
are defined by the property

```math
∫_{-1}^1 ϕ_i(x) ϕ_j(x) dx = 0, \quad \text{for all } i ≠ j, \quad ϕ_0(x) = 1.
```

This property is referred to **orthogonality**. The theoretical benefits of
orthogonality are beyond the scope of this documentation. For more detailed
information about the theoretical properties of the Legendre Polynomials, see
[here](https://en.wikipedia.org/wiki/Legendre_polynomials). It turns out that
Legendre polynomials can be defined by the second order recursive equation

```math
(n + 1) ϕ_{n + 1}(x) = (2 n + 1) x ϕ_n(x) - n ϕ_{n - 1}(x).
```

We prodive two simple methods for evaluating Legendre Polynomials. The simplest
approach is by using [`RecursivePartition.legendre_poly`](@ref).

```jldoctest
using RecursivePartition

legendre_poly(5, 0.5) # 5th order Legendre Polynomial evaluated as x = 0.5

# output

0.08984375
```

However, often the user wants to evaluate the Legendre Polynomials for a range
of values e.g. ``j = 0, \ldots, 5``. In this case, I recommend using the
recursive definintion ([`RecursivePartition.legendre_next`](@ref))
directly.

```jldoctest; setup = :(using RecursivePartition)
lp = Vector{Float64}(undef, 6)
lp[1:2] = [1.0, 0.5]  # Must ϕ_0(x) and ϕ_1(x) as a start.

for j in 3:6 # Remember we are starting from order 0.
  lp[j] = legendre_next(j - 1, 0.5, lp[j - 1], lp[j - 2])
end

lp

# output

6-element Array{Float64,1}:
  1.0
  0.5
 -0.125
 -0.4375
 -0.2890625
  0.08984375
```

## Functions

```@autodocs
Modules = [RecursivePartition]
Pages = ["legendre.jl"]
```
