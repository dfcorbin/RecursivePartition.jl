# Polynomial Chaos Basis

## Definition

The Polynomial Chaos Basis (PCB) is a natural way of extending univariate polynomial
bases to the multivariate setting. I will provide a brief introduction to the
PCB and how it is constructed, but I will make no attempt to motivate its
applications. Upon completion of this documentation I will reference a more
detailed summary of PCB applications and theoretical properties.

Let ``ϕ_j(x)`` denote a univariate polynomial of degree ``j \in \mathbb{N}_{≥ 0 }``
e.g. ``ϕ_2(x) = x^2`` using the Power Basis. We use this notation as there
are other univariate bases to choose from, some of which possess useful
theoretical properties like orthogonality (see [Legendre Polynomials](@ref)). One
way of defining a multivariate polynomial (MVP) is to take a multiplicative combination
of univariate polynomials. This can be indexed using a vector of non-negative
integers as follows. Let ``\boldsymbol{α} = [1, 3, 5]``, the corresponding
3-dimensional multivariate polynomial is given by

```math
  \psi_\boldsymbol{α}(\boldsymbol{x}) = ϕ_1(x_1) \cdot ϕ_3(x_2) \cdot ϕ_5(x_3).
```

The ``d``-dimensional PCB is then defined as the collection of all MVPs
which can be indexed in this way.

```math
  \lbrace ψ_{\boldsymbol{α}} \quad : \quad \boldsymbol{α} \in \mathbb{N}_{≥ 0}^d \rbrace
```

However, in practice this basis must be truncated in order to be used. This can
be achieved by restricting the total degree (defined as the sum of the elements
in ``\boldsymbol{α})``) of the MVPs.

```math
  \lbrace ψ_{\boldsymbol{\alpha}} \quad : \quad \|\boldsymbol{\alpha}\|_1 \leqq J,
  \text{ } \boldsymbol{\alpha} \in \mathbb{N}_{\geqq 0}^d \rbrace.
```

Now suppose you have an ``n \times d`` matrix ``\boldsymbol{X}`` of real numbers.
Each row of ``\boldsymbol{X}`` is a vector on which we can evaluate a
MVP. We can therefore construct a new matrix where the
entry in the ``i``'th row and ``j``'th column is the ``i``'th row of
``\boldsymbol{X}`` evaluated at the ``j``'th basis function in our PCB i.e.
each column of the transformed matrix corresponds to a different MVP in the
truncated PCB. This matrix is what we refer to as the "PCB expansion" of
``\boldsymbol{X}``.

```math
\boldsymbol{Ψ}_{i,j} = ψ_{\boldsymbol{α}_j}(\boldsymbol{x}_i)
```

## Implementation Details

Currently, I have only implemented the [Legendre Polynomials](@ref) variant of
the PCB. Since the Legendre Polynomials are only defined on ``[-1, 1]``, the user
must provide upper/lower bounds for each column of ``\boldsymbol{X}`` (it is
assumed that each column is uniformly distributed between these bounds). These
bounds are used to rescale the columns so that they are all uniformly distributed
on ``[-1, 1]``. Bounds are provided as a ``d \times 2`` matrix. For example, if
``d = 2``, ``x_1 \in [0, 1]`` and ``x_2 \in [1, 2]``, the bounds are represented
as

```math
\begin{pmatrix}
  0 & 1 \\
  1 & 2
\end{pmatrix}
```

Constructing the PCB expansion of a matrix using
[`RecursivePartition.trunc_pcbmat`](@ref) is then straightforward.

```jldoctest
using RecursivePartition

X = [0.0 0.5; -0.5 0.0]
kmat = repeat([-1.0 1.0], 2, 1)

trunc_pcbmat(X, 2, kmat)

# output

2×5 Array{Float64,2}:
 0.5  -0.125   0.0   0.0  -0.5
 0.0  -0.5    -0.5  -0.0  -0.125
```

If there are specific MVPs you wish to use in the basis expansion (as opposed to
the full truncated PCB), this is achieved using
[`RecursivePartition.index_pcbmat`](@ref). This function accepts
a vector of [`RecursivePartition.MVPIndex`](@ref) objects which
correspond to your choice of MVPs. To construct such an object, we supply
two vectors of even length, the first gives the degree of each
multiplicative term in the MVP, the second gives the covariate it applies to.
For example, to store the MVP

```math
ϕ_1(x_1) ⋅ ϕ_3(x_2)
```

we create the object

```julia
MVPIndex([1, 3], [1, 2])
```

This may seem like an odd method for storing an MVP, but it turns out to be
far more memory efficient than other methods in higher dimensions. As an alternative
to manually constructing [`RecursivePartition.MVPIndex`](@ref) objects, you can
generate the full truncated PCB basis using [`RecursivePartition.mvpindex`](@ref)
and select the basis functions you wish to use.

```jldoctest; setup = :(using RecursivePartition)
mvpindex(2, 2) # 2D PCB basis trunacted with J = 2

# output

5-element Array{MVPIndex,1}:
 MVPIndex([1], [2])
 MVPIndex([2], [2])
 MVPIndex([1], [1])
 MVPIndex([1, 1], [1, 2])
 MVPIndex([2], [1])
```

## Functions

```@autodocs
Modules = [RecursivePartition]
Pages = ["pcb.jl"]
```
