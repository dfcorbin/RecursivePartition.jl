function rescale(x::Float64, knots::Vector{Float64})
    if x < knots[1] || x > knots[2]
        throw(DomainError(x, "x uncontained by knots."))
    end
    return 2 * (knots[1] - x) / (knots[1] - knots[2]) - 1
end


function rescale_vec!(x::Vector{Float64}, x1::Vector{Float64},
    kmat::Matrix{Float64})
    for i in 1:length(x)
        x1[i] = rescale(x[i], kmat[i,:])
    end
end


function rescale(x::Vector{Float64}, kmat::Matrix{Float64})
    x1 = Vector{Float64}(undef, length(x))
    rescale_vec!(x, x1, kmat)
    return x1
end


function rescale(X::Matrix{Float64}, kmat::Matrix{Float64})
    X1 = Matrix{Float64}(undef, size(X))
    x1 = Vector{Float64}(undef, size(X, 2))
    for i in 1:size(X, 1)
        rescale_vec!(X[i, :], x1, kmat)
        X1[i,:] .= x1
    end
    return X1
end


function univar_array(X::Matrix{Float64}, degmax::Int64, kmat::Matrix{Float64})
    N, dim = size(X)
    out = Array{Float64, 3}(undef, N, dim, degmax + 1)
    out[:, :, 1] .= 1.0
    rescaled = Vector{Float64}(undef, dim)
    for i in 1:N
        rescale_vec!(X[i, :], rescaled, kmat)
        out[i, :, 2] .= rescaled
    end
    if degmax == 1 return out end
    for obs = 1:N, deg = 3:(degmax + 1), d = 1:dim
        out[obs, d, deg] = legendre_next(deg - 1, out[obs, d, 2],
        out[obs, d, deg - 1], out[obs, d, deg - 2])
    end
    return out
end


"""
    MVPIndex(deg::Vector{Int64}, dim::Vector{Int64})

Compact structure used to index a multivariate polynomial (MVP).

It is assumed that this type is primarily constructed using [`mvpindex`](@ref)
and supplied to [`index_pcbmat`](@ref) to perform a polynomial basis expansion.

- `deg` gives the degree of each multiplicative term in the MVP.
- `dim` gives the covariate associated with each degree.

See also: [`mvpindex`](@ref), [`index_pcbmat`](@ref)

# Examples

Let's suppose we have the vector ``\\boldsymbol{x} = [x_1, x_2]`` and we wish
to index the multivariate polynomial ``ϕ_2(x_1) \\cdot ϕ_2(x)^3``.
This is achieved using the following object.

```julia
MVPIndex([2, 3], [1, 2])
```

Note that ``ϕ_j`` denotes a generic order ``j`` univariate polynomial.
"""
struct MVPIndex
    deg::Vector{Int64}
    dim::Vector{Int64}
end


function MVPIndex()
    deg = Vector{Int64}(undef, 0)
    dim = Vector{Int64}(undef, 0)
    return MVPIndex(deg, dim)
end


function find_mvpindex(d::Int64, degmax::Int64, root::Bool=true,
    d0::Int64=d, index::MVPIndex=MVPIndex(),
    vlist::Vector{MVPIndex} = Vector{MVPIndex}(undef, 0))
    if (degmax == 0) || (d == 0)
        new = deepcopy(index)
        push!(vlist, new)
    else
        for j in 0:degmax
            new = deepcopy(index)
            if j >= 1
                push!(new.deg, j)
                push!(new.dim, d0 - d + 1)
            end
            find_mvpindex(d - 1, degmax - j, false, d0, new, vlist)
        end
    end
    if root
        return vlist[2:end]
    end
end


"""
    mvpindex(dim::Int64, degmax::Int64)

Generate a vector containing all possible [`MVPIndex`](@ref) objects subject to
a bound, `degmax`, on the total degree of each multivariate polynomial.

The total degree of a multivariate polynomial is defined as the sum of each
individual multiplicative term's degree i.e.
``\\text{deg}(x_1^2 x_2^3) = 2 + 3 = 5``

See also: [`index_pcbmat`](@ref)

# Examples

```jldoctest
# Output all 2-dimensional MVPs with total degree less than 2.

mvpindex(2, 2)

# output

5-element Array{MVPIndex,1}:
 MVPIndex([1], [2])
 MVPIndex([2], [2])
 MVPIndex([1], [1])
 MVPIndex([1, 1], [1, 2])
 MVPIndex([2], [1])
```
"""
function mvpindex(dim::Int64, degmax::Int64)
    return find_mvpindex(dim, degmax)
end


function mvp(index::MVPIndex, univar::Array{Float64, 3}, obs::Int64)
    out = 1.0
    for i in 1:length(index.deg)
        out *= univar[obs, index.dim[i], index.deg[i] + 1]
    end
    return out
end


"""
    index_pcbmat(X::Matrix{Float64}, indices::Vector{MVPIndex}, kmat::Matrix{Float64})

Compute the Polynomial Chaos Basis (PCB) expansion for each row of `X` for specified
indices (generated using [`mvpindex`](@ref)).

Legendre Polynomials are used as basis functions, hence an additional argument
`kmat` is supplied to rescale the columns of `X` back on to ``[-1, 1]``.

See also: [`mvpindex`](@ref), [`MVPIndex`](@ref), [`trunc_pcbmat`](@ref)

# Examples

```jldoctest
X = [-0.5 0.5; 0.1 0.2]
kmat = repeat([-1.0 1.0], 2, 1)
ind = mvpindex(2, 2)

index_pcbmat(X, ind, kmat)

# output

2×5 Array{Float64,2}:
 0.5  -0.125  -0.5  -0.25  -0.125
 0.2  -0.44    0.1   0.02  -0.485
```
"""
function index_pcbmat(X::Matrix{Float64}, indices::Vector{MVPIndex},
    kmat::Matrix{Float64})
    N = size(X, 1)
    dim1 = length(indices)
    out = ones(Float64, N, dim1)
    degmax = maximum([maximum(indices[i].deg) for i in 1:dim1])
    univar = univar_array(X, degmax, kmat)
    for col = 1:dim1, row = 1:N
        out[row, col] = mvp(indices[col], univar, row)
    end
    return out
end


"""
    trunc_pcbmat(X::Matrix{Float64}, degmax::Int64, kmat::Matrix{Float64})

Compute the Polynomial Chaos Basis (PCB) using the full truncated PCB basis.

Legendre Polynomials are used as basis functions, hence an additional argument
`kmat` is supplied to rescale the columns of `X` back on to ``[-1, 1]``.

See also: [`mvpindex`](@ref), [`MVPIndex`](@ref), [`index_pcbmat`](@ref)

# Examples

```jldoctest
X = [-0.5 0.5; 0.1 0.2]
kmat = repeat([-1.0 1.0], 2, 1)

trunc_pcbmat(X, 2, kmat)

# output

2×5 Array{Float64,2}:
 0.5  -0.125  -0.5  -0.25  -0.125
 0.2  -0.44    0.1   0.02  -0.485
```
"""
function trunc_pcbmat(X::Matrix{Float64}, degmax::Int64, kmat::Matrix{Float64})
    indices = mvpindex(size(X, 2), degmax)
    return index_pcbmat(X, indices, kmat)
end
