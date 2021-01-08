"""
    legendre_next(ord, x, d1, d0)

Recursive definition of the Legendre polynomials.

The Legendre Polynomials are defined as second order recursive equation,
meaning computing an order ``n > 2`` Legendre Polynomial is possible if we know
the corresponding order ``n - 1`` and ``n - 2`` evaluations. These are supplied
in the arguments `d1` and `d0` (note that the order 1 and 0 evaluations are
always equal to `x` and `1.0` respectively).

See also: [`legendre_poly`](@ref)

# Examples

```jldoctest
ord = 2
x = 1.0
d1 = 1.0
d0 = 1.0

legendre_next(ord, x, d1, d0)

# output

1.0
```
"""
function legendre_next(ord::Int64, x::Float64, d1::Float64, d0::Float64)
    if x < -1.0 || x > 1.0
        throw(DomainError(x, "Legendre polynomials defined on [-1,1]."))
    end
    t1::Float64 = (1 / ord) * (2 * ord - 1) * x * d1
    t2::Float64 = (1 / ord) * (ord - 1) * d0
    return t1 - t2
end

function iter_poly!(ord::Int64, x::Float64, d::Vector{Float64})
    for i in 2:ord
        next = legendre_next(i, x, d[1], d[2])
        d[2] = d[1]
        d[1] = next
    end
end

"""
    legendre_poly(ord, x)

Return the Legendre polynomial of specified order (`ord`) evaluated at `x`.

See also: [`legendre_next`](@ref)

# Examples

```jldoctest
legendre_poly(1, 1.0)

# output

1.0
```
"""
function legendre_poly(ord::Int64, x::Float64)
    if x < -1.0 || x > 1.0
        throw(DomainError(x, "Legendre polynomials defined on [-1,1]."))
    end
    d::Vector{Float64} = [x, 1.0]
    if ord == 0
        return d[2]
    elseif ord == 1
        return d[1]
    else
        iter_poly!(ord, x, d)
    end
    return d[1]
end
