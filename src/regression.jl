"""
    BLMHyper(dim::Int64; shape::Float64=0.001, scale::Float64=0.001)
    BLMHyper(coeff::Vector{Float64}, cov::Matrix{Float64},
        covinv::Matrix{Floa64}, shape::Float64, scale::Float64)

Return container for the hyperparamters associated with a Bayesian linear
model.

If only the dimension/shape/scale parameters are supplied, the default
constructor uses the identity matrix as the covariance matrix, and a vector
of zeros for the the prior mean of the model coefficients.

See also: [`BayesLinearModel`](@ref)
"""
mutable struct BLMHyper
    coeff::Vector{Float64}
    cov::Matrix{Float64}
    covinv::Matrix{Float64}
    shape::Float64
    scale::Float64
    function BLMHyper(coeff, cov, covinv, shape, scale)
        err = ArgumentError("Shape/Scale must be greater than 0.")
        (shape <=0) || (scale <= 0) ? throw(err) : new(coeff, cov, covinv, shape, scale)
    end
end


get_shape(param::BLMHyper) = param.shape
get_scale(param::BLMHyper) = param.scale
get_cov(param::BLMHyper) = param.cov


function BLMHyper(dim::Int64, shape::Float64, scale::Float64)
    coeff = zeros(Float64, dim + 1)
    cov = diagm(ones(Float64, dim + 1))
    return BLMHyper(coeff, cov, cov, shape, scale)
end


"""
Abstract type encompassing different linear models.

See also: [`BayesLinearModel`](@ref), [`PolyBLM`](@ref)
"""
abstract type LinearModel end


"""
    BayesLinearModel(X::Matrix{Float64}, y::Vector{Float64}; shape::Float64=0.001,
            scale::Float64=0.001)
    BayesLinearModel(X::Matrix{Float64}, y::Vector{Float64}, prior::BLMHyper)

Construct [`BayesLinearModel`](@ref) object. One must only supply the dimension
of the linear model, and an optional set of prior hyper parameters. Once the
model is constructed, [`fit!`](@ref) can be used to update it with additional
data.

This object implements the well known Bayesian Linear Model with Gaussian
responses and unknown variance. A Gaussian/inverse-gamma prior is placed on the
model coefficients/variance respectively.

See also: [`BLMHyper`](@ref), [`fit!`](@ref)

# Examples

```jldoctest; setup = :(using Random; Random.seed!(100))
coeff = [1.0, 2.0, 3.0]
f(x) = coeff[1] + coeff[2:3]' * x # Truly linear model
SD = 1.0
X, y = gendat(50000, SD, f, 2)
model = BayesLinearModel(X, y)

get_coeffpost(model)

# output

3-element Array{Float64,1}:
 0.997262172068708
 1.9957265287418318
 3.0063668906195202
```
"""
mutable struct BayesLinearModel <: LinearModel
    prior::BLMHyper
    post::BLMHyper
end


# Accessors for BayesLinearModel abstract type.
get_dim(mod::BayesLinearModel) = length(mod.prior.coeff) - 1
get_scaleprior(mod::BayesLinearModel) = mod.prior.scale
get_shapeprior(mod::BayesLinearModel) = mod.prior.shape
get_covprior(mod::BayesLinearModel) = mod.prior.cov
get_covinvprior(mod::BayesLinearModel) = mod.prior.covinv
get_coeffprior(mod::BayesLinearModel) = mod.prior.coeff
get_scalepost(mod::BayesLinearModel) = mod.post.scale
get_shapepost(mod::BayesLinearModel) = mod.post.shape
get_covpost(mod::BayesLinearModel) = mod.post.cov
get_covinvpost(mod::BayesLinearModel) = mod.post.covinv
get_coeffpost(mod::BayesLinearModel) = mod.post.coeff
get_N(mod::BayesLinearModel) = Int64(2 * (get_shapepost(mod) - get_shapeprior(mod)))

# Setters for BayesLinearModel
set_scaleprior!(mod::BayesLinearModel, val) = begin mod.prior.scale = val end
set_shapeprior!(mod::BayesLinearModel, val) = begin mod.prior.shape = val end
set_covprior!(mod::BayesLinearModel, val) = begin mod.prior.cov = val end
set_covinvprior!(mod::BayesLinearModel, val) = begin mod.prior.covinv = val end
set_coeffprior!(mod::BayesLinearModel, val) = begin mod.prior.coeff = val end
set_scalepost!(mod::BayesLinearModel, val) = begin mod.post.scale = val end
set_shapepost!(mod::BayesLinearModel, val) = begin mod.post.shape = val end
set_covpost!(mod::BayesLinearModel, val) = begin mod.post.cov = val end
set_covinvpost!(mod::BayesLinearModel, val) = begin mod.post.covinv = val end
set_coeffpost!(mod::BayesLinearModel, val) = begin mod.post.coeff = val end


function set_all_post!(mod::BayesLinearModel, coeff::Vector{Float64},
        cov::Matrix{Float64}, covinv::Matrix{Float64}, shape::Float64,
    scale::Float64)
    set_coeffpost!(mod, coeff)
    set_covpost!(mod, cov)
    set_covinvpost!(mod, covinv)
    set_shapepost!(mod, shape)
    set_scalepost!(mod, scale)
end


"""
    fit!(mod::BayesLinearModel, X::Matrix{Float64}, y::Vector{Float64})

Update a Bayesian Linear Model object with data matrix and response variables.

See also: [`BayesLinearModel`](@ref)

# Examples

```jldoctest; setup = :(using Random; Random.seed!(100))
coeff = [1.0, 2.0, 3.0]
f(x) = coeff[1] + coeff[2:3]' * x # Truly linear model
SD = 1.0
X1, y1 = gendat(25000, SD, f, 2)
X2, y2 = gendat(25000, SD, f, 2)
model = BayesLinearModel(X1, y1) # Fit initial model
fit!(model, X2, y2) # Update model with more data

get_coeffpost(model)

# output

3-element Array{Float64,1}:
 0.996650221433608
 1.9907052757630952
 2.9924622031151826
```
"""
function fit!(mod::BayesLinearModel, X::Matrix{Float64}, y::Vector{Float64})
    if length(y) != size(X, 1)
        throw(DimensionMismatch("Must have response for every predictor."))
    end
    X1 = hcat(ones(size(X, 1)), X)
    covinv1::Matrix{Float64} = X1' * X1 + get_covinvpost(mod)
    cov1::Matrix{Float64} = inv(covinv1)
    coeff1::Vector{Float64} = cov1 * (get_covinvpost(mod) * get_coeffpost(mod) + X1' * y)
    shape1::Float64 = get_shapepost(mod) + length(y) / 2
    scale1::Float64 = get_scalepost(mod) + 0.5 * (
        y' * y - coeff1' * covinv1 * coeff1 +
        get_coeffpost(mod)' * get_covinvpost(mod) * get_coeffpost(mod)
    )
    set_all_post!(mod, coeff1, cov1, covinv1, shape1, scale1)
end


function BayesLinearModel(X::Matrix{Float64}, y::Vector{Float64}, prior::BLMHyper)
    mod = BayesLinearModel(prior, deepcopy(prior))
    fit!(mod, X, y)
    return mod
end


function BayesLinearModel(X::Matrix{Float64}, y::Vector{Float64};
        shape::Float64=0.001, scale::Float64=0.001)
    dim = size(X, 2)
    prior = BLMHyper(dim, shape, scale)
    return BayesLinearModel(X, y, prior)
end


function fit!(mod::BayesLinearModel, x::Vector{Float64}, y::Float64)
    x1 = [1.0, x...]
    covinv1::Matrix{Float64} = x1 * x1' + get_covinvpost(mod)
    cov1 = woodbury_inv(get_covpost(mod), x1, x1)
    coeff1::Vector{Float64} = cov1 * (get_covinvpost(mod) * get_coeffpost(mod) + x1 * y)
    shape1::Float64 = get_shapepost(mod) + 0.5
    scale1::Float64 = get_scalepost(mod) + 0.5 * (
        y * y - coeff1' * covinv1 * coeff1 +
        get_coeffpost(mod)' * get_covinvpost(mod) * get_coeffpost(mod)
    )
    set_all_post!(mod, coeff1, cov1, covinv1, shape1, scale1)
end


"""
    predict(mod::BayesLinearModel, X::Matrix{Float64})

Predict responses based on data matrix `X`.
```
"""
function predict(mod::BayesLinearModel, X::Matrix{Float64})
    return [ones(Float64, size(X, 1)) X] * get_coeffpost(mod)
end


"""
    predfun(mod::LinearModel)

Return a function which predicts response variables given an input vector
`x`.
"""
function predfun(mod::BayesLinearModel)
    coeff = get_coeffpost(mod)
    function f(x::Vector{Float64})::Float64
        return [1.0, x...]' * coeff
    end
    return f
end


"""
    logevidence(mod::LinearModel)

Compute the Bayesian Model Evidence/Marginal Likelihood.
"""
function logevidence(mod::BayesLinearModel)
    N = get_N(mod)
    sh0, sh1 = get_shapeprior(mod), get_shapepost(mod)
    sc0, sc1 = get_scaleprior(mod), get_scalepost(mod)
    cov0, cov1 = get_covprior(mod), get_covpost(mod)
    if N <= 0
        throw(ArgumentError("Attempted to compute evidence without data."))
    end
    t1 = - N / 2 * log(2 * pi)
    t2 = sh0 * log(sc0) - sh1 * log(sc1)
    t3 = loggamma(sh1) - loggamma(sh0)
    t4 = 0.5 * (logdet(cov1) - logdet(cov0))
    return +(t1, t2, t3, t4)
end


"""
    PolyBLM(X, y, degmax, bounds; maxparam=200, shape=0.001, scale0.001,
        priorgen=identity_hyper)

Construct a linear model using features derrived from the [Polynomial Chaos
Basis](@ref).

A bound on the maximum number of model parameters, `maxparam`, is specified by
the user. If the number of parameters exceeds this bound, the LARS algorithm is
used to choose the "best" set of parameters satisfying the bound.

Since we do not know which/how many parameters will be included in the model
(we only that the number of parameters is bounded by `maxparam`), it is not
possible to supply an object of type [`BLMHyper`](@ref) as a prior distribution.
Instead, the argument `priorgen` accepts a function which specifies how the
prior distribution should be generated. `priorgen` **must** have the signature

    priorgen(indices::Vector{MVPIndex}, shape::Float64, scale::Float64)

and return an object of type [`BLMHyper`](@ref)

See also: [`BayesLinearModel`](@ref), [`BLMHyper`](@ref), [`MVPIndex`](@ref)
"""
mutable struct PolyBLM <: LinearModel
    blm::BayesLinearModel
    indices::Vector{MVPIndex}
    kmat::Matrix{Float64}
end


# Accessors for PolyBLM abstract type.
get_dim(mod::PolyBLM) = get_dim(mod.blm)
get_scaleprior(mod::PolyBLM) = get_scaleprior(mod.blm)
get_shapeprior(mod::PolyBLM) = get_shapeprior(mod.blm)
get_covprior(mod::PolyBLM) = get_covprior(mod.blm)
get_covinvprior(mod::PolyBLM) = get_covinvprior(mod.blm)
get_coeffprior(mod::PolyBLM) = get_coeffprior(mod.blm)
get_scalepost(mod::PolyBLM) = get_scalepost(mod.blm)
get_shapepost(mod::PolyBLM) = get_shapepost(mod.blm)
get_covpost(mod::PolyBLM) = get_covpost(mod.blm)
get_covinvpost(mod::PolyBLM) = get_covinvpost(mod.blm)
get_coeffpost(mod::PolyBLM) = get_coeffpost(mod.blm)
get_N(mod::PolyBLM) = Int64(2 * (get_shapepost(mod) - get_shapeprior(mod)))
get_kmat(mod::PolyBLM) = mod.kmat
get_indices(mod::PolyBLM) = mod.indices
get_blm(mod::PolyBLM) = mod.blm

# Setters for PolyBLM
set_scaleprior!(mod::PolyBLM, val) = set_scaleprior!(mod.blm, val)
set_shapeprior!(mod::PolyBLM, val) = set_shapeprior!(mod.blm, val)
set_covprior!(mod::PolyBLM, val) = set_covprior!(mod.blm, val)
set_covinvprior!(mod::PolyBLM, val) = set_covinvprior!(mod.blm, val)
set_coeffprior!(mod::PolyBLM, val) = set_coeffprior!(mod.blm, val)
set_scalepost!(mod::PolyBLM, val) = set_scalepost!(mod.blm, val)
set_shapepost!(mod::PolyBLM, val) = set_shapepost!(mod.blm, val)
set_covpost!(mod::PolyBLM, val) = set_covpost!(mod.blm, val)
set_covinvpost!(mod::PolyBLM, val) = set_covinvpost!(mod.blm, val)
set_coeffpost!(mod::PolyBLM, val) = set_coeffpost!(mod.blm, val)


function polymod_pcbmat(mod::PolyBLM, X::Matrix{Float64}; intercept::Bool=false)
    indices = get_indices(mod)
    kmat = get_kmat(mod)
    modmat = index_pcbmat(X, indices, kmat)
    if intercept
        inter = ones(Float64, size(X, 1))
        modmat = hcat(inter, modmat)
    end
    return modmat
end


function polymod_pcbmat(mod::PolyBLM, x::Vector{Float64}; intercept::Bool=false)
    x1 = reshape(x, (1,:))
    modmat = polymod_pcbmat(mod, x1; intercept=intercept)
    modmat1 = reshape(modmat, (:))
    return modmat1
end


function boundedvar(X::Matrix{Float64}, y::Vector{Float64},
    indices::Vector{MVPIndex}, kmat::Matrix{Float64}, maxparam::Int64)
    modmat = index_pcbmat(X, indices, kmat)
    if length(indices) <= maxparam return modmat, indices end
    lasso = glmnet(modmat, y).betas
    var = Vector{Bool}(undef, length(indices))
    for j in 1:size(lasso, 2)
        if norm(lasso[:, j], 0) > maxparam
            var = (lasso[:, j - 1] .!= 0)
            break
        end
    end
    return modmat[:, var], indices[var]
end


function varselect(X::Matrix{Float64}, y::Vector{Float64},
    indices::Vector{MVPIndex}, kmat::Matrix{Float64}, maxparam::Int64)
    if maxparam == 0
        modmat = index_pcbmat(X, indices, kmat)
        return (modmat, indices)
    elseif maxparam > 0
        return boundedvar(X, y, indices, kmat, maxparam)
    else
        throw(ArgumentError("maxparam must be non-negative."))
    end
end


function identity_hyper(indices::Vector{MVPIndex}, shape::Float64,
    scale::Float64)
    dim = length(indices)
    coeff = zeros(Float64, dim + 1)
    cov = diagm(ones(Float64, dim + 1))
    return BLMHyper(coeff, cov, cov, shape, scale)
end


function PolyBLM(X::Matrix{Float64}, y::Vector{Float64}, degmax::Int64,
    bounds::Matrix{Float64}; maxparam::Int64=200, shape::Float64=0.001,
    scale::Float64=0.001, priorgen::Function=identity_hyper)
    dim = size(X, 2)
    indices = mvpindex(dim, degmax)
    # println(bounds)
    # println(maximum(X[1,:]))
    modmat, indices = varselect(X, y, indices, bounds, maxparam)
    prior = priorgen(indices, shape, scale)
    blm = BayesLinearModel(modmat, y, prior)
    return PolyBLM(blm, indices, bounds)
end


function PolyBLM(X::Matrix{Float64}, y::Vector{Float64}, degmax::Int64,
    bounds::Vector{Float64}; maxparam::Int64=200)
    kmat = repeat([bounds[1] bounds[2]], size(X, 2), 1)
    return PolyBLM(X, y, degmax, kmat; maxparam=maxparam)
end


function fit!(mod::PolyBLM, X::Matrix{Float64}, y::Vector{Float64})
    modmat = polymod_pcbmat(mod, X)
    fit!(get_blm(mod), modmat, y)
end


function predict(mod::PolyBLM, X::Matrix{Float64})
    modmat = polymod_pcbmat(mod, X)
    predict(get_blm(mod), modmat)
end


function predfun(mod::PolyBLM)
    p = predfun(get_blm(mod))
    function f(x::Vector{Float64})::Float64
        x1 = reshape(x, (1, :))
        ϕ = [1.0, polymod_pcbmat(mod, x1)...]
        return ϕ' * get_coeffpost(mod)
    end
    return f
end


logevidence(mod::PolyBLM) = logevidence(get_blm(mod))
