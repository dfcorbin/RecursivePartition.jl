"""
    splitmat(mat::Matrix{Float64}, dim::Int64, loc::Float64)

Return two new matrices that result from splitting row `dim` at `loc`.

`mat` must have exactly two columns, with column two being strictly greater
than column 1. `loc` must be contained by the row to be split.
"""
function splitmat(mat::Matrix{Float64}, dim::Int64, loc::Float64)
    if size(mat, 2) != 2
        throw(ArgumentError("mat must have exactly two columns."))
    elseif sum(mat[:,1] .< mat[:,2]) != size(mat, 1)
        throw(ArgumentError("Left col of knotmat must be < right col."))
    elseif (loc <= mat[dim, 1]) || (loc >= mat[dim,2])
        throw(DomainError(loc, "uncontained knot."))
    end
    k1 = deepcopy(mat)
    k2 = deepcopy(mat)
    k1[dim,2] = loc
    k2[dim,1] = loc
    return (k1, k2)
end

"""
    insert_knot!(P::Matvec{Float64}, k::Int64, dim::Int64, loc::Float64)

Do [`insert_knot`](@ref) by overwriting the original array (memory efficient).
"""
function insert_knot!(P::Matvec{Float64}, k::Int64, dim::Int64,
    loc::Float64)
    s1, s2 = splitmat(P[k], dim, loc)
    P[k] = s1
    push!(P, s2)
end

"""
    insert_knot(P::Matvec{Float64}, k::Int64, dim::Int64, loc::Float64)

Bisect the `k`'th subset along dimension `dim`.

This function returns a modified version of `P`, where the `k`'th subset is
replaced by the two matrices outputted by [`splitmat`](@ref). The left matrix
of the split is inserted into the `k`'th position, the right side is appended
to the end.

See also: [`splitmat`](@ref), [`insert_knot!`](@ref)

# Examples

```jldoctest
P = [repeat([-1.0 1.0], 2, 1)]
P1 = insert_knot(P, 1, 1, 0.0)

# output

2-element Array{Array{Float64,2},1}:
 [-1.0 0.0; -1.0 1.0]
 [0.0 1.0; -1.0 1.0]
```
"""
function insert_knot(P::Matvec{Float64}, k::Int64, dim::Int64,
    loc::Float64)
    P1 = deepcopy(P)
    insert_knot!(P1, k, dim, loc)
    return P1
end


function is_contained(x::Vector{Float64}, knotmat::Matrix{Float64},
    upper::Vector{Float64})
    if length(x) != size(knotmat, 1)
        throw(ArgumentError("Must have single interval for every dimension."))
    end
    for j in 1:length(x)
        if x[j] < knotmat[j,1] return false end
        if x[j] == upper[j]
            if x[j] > knotmat[j,2] return false end
        else
            if x[j] >= knotmat[j,2] return false end
        end
    end
    return true
end


function get_upper(P::Matvec{Float64})
    dim = size(P[1], 1)
    upper = P[1][:, 2]
    for k in 2:length(P), d in 1:dim
        upper[d] = max(upper[d], P[k][d, 2])
    end
    return upper
end


function is_contained(X::Matrix{Float64}, knotmat::Matrix{Float64},
    upper::Vector{Float64})
    rows = Vector{Bool}(undef, size(X, 1))
    for i in 1:size(X, 1)
        rows[i] = is_contained(X[i,:], knotmat, upper)
    end
    return rows
end


function which_subset(x::Vector{Float64}, P::Matvec{Float64},
    upper::Vector{Float64})
    for i in 1:length(P)
        if is_contained(x, P[i], upper) return i end
    end
    throw(DomainError("Partition does not contain all observations."))
end


"""
    which_subset(x::Vector{Float64}, P::Matvec{Float64})

Determine which subset a vector `x` is contained by. This function assumes
that your partition `P` is disjoint and comprises the entire space under union.

# Examples

```jldoctest
P = [repeat([-1.0 1.0], 2, 1)]
insert_knot!(P, 1, 1, 0.0) # Create two daughter subsets by dividing dim 1.
x = [-0.5, 0.0] # Dim 1 is les than 0.

which_subset(x, P)

# output

1
```
"""
function which_subset(x::Vector{Float64}, P::Matvec{Float64})
    upper = get_upper(P)
    return which_subset(x, P, upper)
end


"""
    partition(X, P, [, y]; track=false)

Partition a data matrix X (and optional vector y) into subsets according to a
partition P.

See also: [`which_subset`](@ref)

# Examples

```jldoctest
P = [repeat([-1.0 1.0], 2, 1)]
P = insert_knot!(P, 1, 1, 0.0)
X = [-0.5 0.0; 0.5 0.0]
y = [1.0, 2.0]
partition(X, P, y; track=true)

# output

([[-0.5 0.0], [0.5 0.0]], [[1.0], [2.0]], [[1], [2]])
```
"""
function partition(X::Matrix{Float64}, P::Matvec{Float64},
    y::Vector{Float64}, upper::Vector{Float64}; track=false)
    if length(y) != size(X, 1)
        throw(ArgumentError("length of y must be equal to #row X."))
    end
    X1 = deepcopy(X)
    y1 = deepcopy(y)
    K = length(P)
    Xsubsets = Matvec{Float64}(undef, K)
    ysubsets = Vecvec{Float64}(undef, K)
    if track
        rows = Vecvec{Int64}(undef, K)
        rowstmp = [1:length(y)...]
    end
    for i in 1:K
        r = is_contained(X1, P[i], upper)
        Xsubsets[i] = X1[r, :]
        ysubsets[i] = y1[r]
        if track
            rows[i] = rowstmp[r]
            rowstmp = rowstmp[.!r]
        end
        X1 = X1[.!r,:]
        y1 = y1[.!r]
    end
    return track ? (Xsubsets, ysubsets, rows) : (Xsubsets, ysubsets)
end


function partition(X::Matrix{Float64}, P::Matvec{Float64},
    y::Vector{Float64}; track=false)
    upper = get_upper(P)
    return partition(X, P, y, upper; track=track)
end


function partition(X::Matrix{Float64}, P::Matvec{Float64}; track=false)
    upper = get_upper(P)
    X1 = deepcopy(X)
    K = length(P)
    if track
        rows = Vecvec{Int64}(undef, K)
        rowstmp = [1:size(X, 1)...]
    end
    Xsubsets = Matvec{Float64}(undef, K)
    for i in 1:K
        r = is_contained(X1, P[i], upper)
        Xsubsets[i] = X1[r, :]
        if track
            rows[i] = rowstmp[r]
            rowstmp = rowstmp[.!r]
        end
        X1 = X1[.!r,:]
    end
    return track ? (Xsubsets, rows) : Xsubsets
end
