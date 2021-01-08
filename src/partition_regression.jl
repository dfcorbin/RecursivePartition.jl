## PartitionModel code...


abstract type ModArgs end


mutable struct PartitionHyper
    shape::Float64
    scale::Float64
    logdet::Union{Nothing, Vector{Float64}}
end


mutable struct PartitionModel{T <: LinearModel}
    X::Matvec
    y::Vecvec
    P::Matvec
    modvec::Vector{T}
    gammaprior::PartitionHyper
    gammapost::PartitionHyper
    logev::Union{Nothing, Float64}
end


get_loc_X(mod::PartitionModel, k::Int64) = mod.X[k]
get_loc_y(mod::PartitionModel, k::Int64) = mod.y[k]
get_loc_coeffpost(mod::PartitionModel, k::Int64) = mod.modvec[k].post.coeff
get_P(mod::PartitionModel) = mod.P
get_K(mod::PartitionModel) = length(mod.P)
get_lm(mod::PartitionModel, k::Int64) = mod.modvec[k]
get_logev(mod::PartitionModel) = mod.logev
get_modvec(mod::PartitionModel) = mod.modvec

get_scaleprior(mod::PartitionModel) = mod.gammaprior.scale
get_shapeprior(mod::PartitionModel) = mod.gammaprior.shape
get_covprior(mod::PartitionModel, k::Int64) = mod.modvec[k].prior.cov
get_logdetprior(mod::PartitionModel) = mod.gammaprior.logdet

get_scalepost(mod::PartitionModel) = mod.gammapost.scale
get_shapepost(mod::PartitionModel) = mod.gammapost.shape
get_covpost(mod::PartitionModel, k::Int64) = mod.modvec[k].post.cov
get_logdetpost(mod::PartitionModel) = mod.gammapost.logdet

set_logdetprior!(mod::PartitionModel, k::Int64, val::Float64) = begin mod.gammaprior.logdet[k] = val end
set_logdetprior!(mod::PartitionModel, val) = begin mod.gammaprior.logdet = val end
set_scalepost!(mod::PartitionModel, val::Float64) = begin mod.gammapost.scale = val end
set_logdetpost!(mod::PartitionModel, k::Int64, val::Float64) = begin mod.gammapost.logdet[k] = val end
set_logdetpost!(mod::PartitionModel, val) = begin mod.gammapost.logdet = val end
set_shapepost!(mod::PartitionModel, val::Float64) = begin mod.gammapost.shape = val end
set_logev!(mod::PartitionModel, val) = begin mod.logev = val end
set_lm!(mod::PartitionModel, k::Int64, val::LinearModel) = begin mod.modvec[k] = val end
append_modvec!(mod::PartitionModel, val::LinearModel) = begin push!(mod.modvec, val) end


function append_logdetprior!(mod::PartitionModel, val::Float64)
    push!(mod.gammaprior.logdet, val)
end


function append_logdetpost!(mod::PartitionModel, val::Float64)
    push!(mod.gammapost.logdet, val)
end


function logevidence(priorlogdet, postlogdet, shape, scale, shape0, scale0, N)
    t1 = - N / 2 * log(2 * pi)
    t2 = shape0 * log(scale0) - shape * log(scale)
    t3 = loggamma(shape) - loggamma(shape0)
    t4 = 0.5 * sum(postlogdet .- priorlogdet)
    return +(t1, t2, t3, t4)
end


function setup_PartitionModel(Xvec::Matvec, yvec::Vecvec, P::Matvec,
        modvec::Vector{<: LinearModel}, N::Int64, K::Int64, shape::Float64,
        scale::Float64)
    logdetprior = [logdet(get_covprior(modvec[k])) for k in 1:K]
    logdetpost = [logdet(get_covpost(modvec[k])) for k in 1:K]
    shapepost = shape + N / 2
    scalepost = scale + sum([get_scalepost(modvec[k]) - scale for k in 1:K])
    gammaprior = PartitionHyper(shape, scale, logdetprior)
    gammapost = PartitionHyper(shapepost, scalepost, logdetpost)
    logev = logevidence(logdetprior, logdetpost, shapepost, scalepost, shape,
    scale, N)
    return PartitionModel(Xvec, yvec, P, modvec, gammaprior, gammapost, logev)
end


function PartitionModel{T}(X::Matrix{Float64}, y::Vector{Float64},
        P::Matvec, args::ModArgs) where T <: LinearModel
    K = length(P)
    dim = size(P[1], 1)
    Xvec, yvec = partition(X, P, y)
    if [size(P[k]) == (dim, 2) for k in 1:K] != [true for k in 1:K]
        throw(ArgumentError("Partition subsets must have `dim` rows/2 columns."))
    end
    modvec = construct_modvec(Xvec, yvec, args, P)
    return setup_PartitionModel(Xvec, yvec, P, modvec, length(y), K, get_shape(args),
    get_scale(args))
end


function partition_polyblm(X::Matrix{Float64}, y::Vector{Float64},
        P::Matvec; degmax::Int64=3, maxparam::Int64=200,
        priorgen::Function=identity_hyper, shape::Float64=0.001,
        scale::Float64=0.001)
    args = PolyArgs(degmax, maxparam, priorgen, shape, scale)
    return PartitionModel{PolyBLM}(X, y, P, args)
end


function partition_blm(X::Matrix{Float64}, y::Vector{Float64},
        P::Matvec, prior::BLMHyper)
    args = BLMArgs(prior)
    return PartitionModel{BayesLinearModel}(X, y, P, args)
end


function partition_blm(X::Matrix{Float64}, y::Vector{Float64},
        P::Matvec; shape::Float64=0.001, scale::Float64=0.001)
    dim = size(X, 2)
    prior = BLMHyper(dim, shape, scale)
    return partition_blm(X, y, P, prior)
end


mutable struct SubsetMem{T <: LinearModel}
    X::Matrix{Float64}
    y::Vector{Float64}
    model::T
    kmat::Matrix{Float64}
end


get_kmat(s::SubsetMem) = s.kmat


function search_storage(subsets::Matvec, stored::Vector{SubsetMem{T}}) where T <: LinearModel
    if length(subsets) != 2
        error = ArgumentError("Must provide exactly two reference subsets.")
        throw(error)
    end
    pos  = zeros(Int64, 2)
    if length(stored) == 0 return pos end
    located = 0
    for k in 1:length(stored)
        if subsets[1] == get_kmat(stored[k])
            pos[1] = k
            located += 1
        elseif subsets[2] == get_kmat(stored[k])
            pos[2] = k
            located += 1
        end
        if located == 2 break end
    end
    return pos
end


function add_subsets_from_mem!(mod::PartitionModel{T}, stored::Vector{SubsetMem{T}},
        pos::Vector{Int64}, k::Int64) where T <: LinearModel
    mod.X[k] = stored[pos[1]].X
    push!(mod.X, stored[pos[2]].X)
    mod.y[k] = stored[pos[1]].y
    push!(mod.y, stored[pos[2]].y)
    mod.modvec[k] = stored[pos[1]].model
    push!(mod.modvec, stored[pos[2]].model)
end


function update_data!(mod::PartitionModel, k::Int64, mindat::Int64)
    Xvec, yvec = partition(mod.X[k], [mod.P[k], mod.P[end]], mod.y[k])
    size_l = length(yvec[1]); size_r = length(yvec[2])
    if (size_l <= mindat) || (size_r <= mindat)
        return mod = nothing
    end
    mod.X[k] = Xvec[1]
    push!(mod.X, Xvec[2])
    mod.y[k] = yvec[1]
    push!(mod.y, yvec[2])
    return true
end


function update_storage!(mod::PartitionModel, k::Int64,
        stored::Vector{SubsetMem{T}}) where T <: LinearModel
    left = SubsetMem(mod.X[k], mod.y[k], mod.modvec[k], mod.P[k])
    right = SubsetMem(mod.X[end], mod.y[end], mod.modvec[end], mod.P[end])
    push!(stored, left, right)
end


function update_scale!(mod::PartitionModel, parent_scale::Float64, k::Int64)
    K = get_K(mod)
    new = get_scalepost(mod) -
        (parent_scale - get_scaleprior(mod)) + # Remove parent subset contribution.
        (get_loc_scalepost(mod, k) - get_scaleprior(mod)) +
        (get_loc_scalepost(mod, K) - get_scaleprior(mod))
    set_scalepost!(mod, new)
end


function update_dets!(mod::PartitionModel, k::Int64)
    K = get_K(mod)
    set_logdetprior!(mod, k, logdet(get_covprior(mod, k)))
    append_logdetprior!(mod, logdet(get_covprior(mod, K)))
    set_logdetpost!(mod, k, logdet(get_covpost(mod, k)))
    append_logdetpost!(mod, logdet(get_covpost(mod, K)))
end


function update_logev!(mod::PartitionModel, k::Int64, prev_scale::Float64,
        parent_priorlogdet::Float64, parent_postlogdet::Float64)
    mod.logev = mod.logev + mod.gammapost.shape * log(prev_scale) -
              mod.gammapost.shape * log(mod.gammapost.scale) -
              0.5 * (parent_postlogdet - parent_priorlogdet) +
              0.5 * (mod.gammapost.logdet[k] - mod.gammaprior.logdet[k]) +
              0.5 * (mod.gammapost.logdet[end] - mod.gammaprior.logdet[end])
end


function split_subset(mod::PartitionModel{T}, stored::Vector{SubsetMem{T}},
        k::Int64, dim::Int64, loc::Float64, mindat::Int64,
        args::ModArgs) where T <: LinearModel
    mod1 = deepcopy(mod)
    insert_knot!(mod1.P, k, dim, loc)
    pos = search_storage([mod1.P[k], mod1.P[end]], stored)
    parent_scale = get_loc_scalepost(mod1, k)
    prev_scale = mod1.gammapost.scale
    parent_priorlogdet = mod1.gammaprior.logdet[k]
    parent_postlogdet = mod1.gammapost.logdet[k]
    if pos != [0, 0]
        add_subsets_from_mem!(mod1, stored, pos, k)
    else
        update_data!(mod1, k, mindat)
        if isnothing(mod1) return nothing end
        update_models!(mod1, k, args)
        update_storage!(mod1, k, stored)
    end
    update_scale!(mod1, parent_scale, k)
    update_dets!(mod1, k)
    update_logev!(mod1, k, prev_scale, parent_priorlogdet, parent_postlogdet)
    return mod1
end


## BayesLinearModel specific implementations.


# Getters and setters.
get_loc_scalepost(mod::PartitionModel{BayesLinearModel}, k::Int64) = mod.modvec[k].post.scale
get_covprior(mod::PartitionModel{BayesLinearModel}, k::Int64) = mod.modvec[k].prior.cov
get_covpost(mod::PartitionModel{BayesLinearModel}, k::Int64) = mod.modvec[k].post.cov


struct BLMArgs <: ModArgs
    prior::BLMHyper
end


get_shape(args::BLMArgs) = args.prior.shape
get_scale(args::BLMArgs) = args.prior.scale
get_prior(args::BLMArgs) = args.prior


function construct_modvec(Xvec::Matvec, yvec::Vecvec, args::BLMArgs, P::Matvec)
    modvec = Vector{BayesLinearModel}(undef, length(P))
    for k in 1:length(P)
        prior = get_prior(args)
        modvec[k] = BayesLinearModel(Xvec[k], yvec[k], prior)
    end
    return modvec
end


function update_models!(mod::PartitionModel{BayesLinearModel}, k::Int64,
    args::BLMArgs)
    K = get_K(mod)
    Xl, Xr = get_loc_X(mod, k), get_loc_X(mod, K)
    yl, yr = get_loc_y(mod, k), get_loc_y(mod, K)
    prior = get_prior(args)
    leftmod = BayesLinearModel(Xl, yl, prior)
    rightmod = BayesLinearModel(Xr, yr, prior)
    set_lm!(mod, k, leftmod)
    append_modvec!(mod, rightmod)
end


## PolyBLM specific implementations

# Getters and setters for PolyBLM.
get_loc_scalepost(mod::PartitionModel{PolyBLM}, k::Int64) = mod.modvec[k].blm.post.scale
get_covprior(mod::PartitionModel{PolyBLM}, k::Int64) = mod.modvec[k].blm.prior.cov
get_covpost(mod::PartitionModel{PolyBLM}, k::Int64) = mod.modvec[k].blm.post.cov


struct PolyArgs <: ModArgs
    degmax::Int64
    maxparam::Int64
    priorgen::Function
    shape::Float64
    scale::Float64
end

get_degmax(args::PolyArgs) = args.degmax
get_maxparam(args::PolyArgs) = args.maxparam
get_priorgen(args::PolyArgs) = args.priorgen
get_shape(args::PolyArgs) = args.shape
get_scale(args::PolyArgs) = args.scale


function construct_modvec(Xvec::Matvec, yvec::Vecvec, args::PolyArgs, P::Matvec)
    modvec = Vector{PolyBLM}(undef, length(P))
    for k in 1:length(P)
        modvec[k] = PolyBLM(Xvec[k], yvec[k], get_degmax(args), P[k];
        maxparam=get_maxparam(args), shape=get_shape(args), scale=get_scale(args),
        priorgen=get_priorgen(args))
    end
    return modvec
end


function update_models!(mod::PartitionModel{PolyBLM}, k::Int64, args::PolyArgs)
    K = length(get_P(mod))
    leftmod = PolyBLM(get_loc_X(mod, k), get_loc_y(mod, k),
        get_degmax(args), get_P(mod)[k]; maxparam=get_maxparam(args), shape=get_shapeprior(mod),
        scale=get_scaleprior(mod), priorgen=get_priorgen(args))
    rightmod = PolyBLM(get_loc_X(mod, K), get_loc_y(mod, K), get_degmax(args),
        get_P(mod)[K]; maxparam=get_maxparam(args), shape=get_shapeprior(mod),
        scale=get_scaleprior(mod), priorgen=get_priorgen(args))
    set_lm!(mod, k, leftmod)
    append_modvec!(mod, rightmod)
end


## Auto partitioning funciton


function best_dim_split(mod::PartitionModel{T}, stored::Vector{SubsetMem{T}},
    k::Int64, mindat::Int64, args::ModArgs, verbose) where T <: LinearModel
    dim = size(get_loc_X(mod, k), 2)
    kmat = get_P(mod)[k]
    proposed = Vector{PartitionModel{T}}(undef, dim)
    ev_vals = Vector{Float64}(undef, dim)
    if verbose print("    fitting dim ") end
    for d in 1:dim
        if verbose print(d, ", ") end
        loc = (kmat[d, 1] + kmat[d, 2]) / 2
        proposed[d] = split_subset(mod, stored, k, d, loc, mindat, args)
        ev_vals[d] = get_logev(proposed[d])
    end
    best_mod = findmax(ev_vals)[2]
    if verbose println("\n    dimension ", best_mod, " trialed...") end
    return proposed[best_mod]
end


function auto_partition_model(X::Matrix{Float64}, y::Vector{Float64},
    bounds::Matrix{Float64}, args::ModArgs, T::Type, mindat::Union{Nothing, Int64},
    Kmax::Int64, verbose)
    P = [bounds]
    K = 1
    mod = PartitionModel{T}(X, y, P, args)
    stored = Vector{SubsetMem{T}}(undef, 0)
    while K < Kmax
        split = false
        for k in randperm(K)
            if verbose println("\n\nTesting subset ", k, ":") end
            modnew = best_dim_split(mod, stored, k, mindat, args, verbose)
            if isnothing(modnew) continue end
            if get_logev(modnew) > get_logev(mod) # Split led to improvement.
                if verbose println("    SPLIT ACCEPTED!") end
                mod = modnew
                split = true
                K += 1
                if K >= Kmax break end
            end
        end
        if !split break end
    end
    return mod
end


function auto_partition_polyblm(X::Matrix{Float64}, y::Vector{Float64},
        bounds::Union{Matrix{Float64}, Vector{Float64}}; degmax::Int64=3,
        maxparam::Int64=200, priorgen::Function=identity_hyper,
        shape::Float64=0.001, scale::Float64=0.001, mindat=nothing, Kmax=200,
        verbose=false)
    dim = size(X, 2)
    if typeof(bounds) == Vector{Float64}
        b1 = reshape(bounds, (1, 2))
        bounds1 = repeat(b1, dim, 1)
    end
    if isnothing(mindat) mindat = 2 * length(mvpindex(dim, degmax)) end
    args = PolyArgs(degmax, maxparam, priorgen, shape, scale)
    return auto_partition_model(X, y, bounds1, args, PolyBLM, mindat, Kmax,
        verbose)
end


function auto_partition_blm(X::Matrix{Float64}, y::Vector{Float64},
        bounds::Union{Matrix{Float64}, Vector{Float64}}, prior::BLMHyper;
        mindat=nothing, Kmax=200, verbose=false)
    dim = size(X, 2)
    if typeof(bounds) == Vector{Float64}
        b1 = reshape(bounds, (1, 2))
        bounds1 = repeat(b1, dim, 1)
    end
    if isnothing(mindat) mindat = 2 * dim end
    args = BLMArgs(prior)
    return auto_partition_model(X, y, bounds1, args, BayesLinearModel, mindat, Kmax,
        verbose)
end


function auto_partition_blm(X::Matrix{Float64}, y::Vector{Float64},
        bounds::Union{Matrix{Float64}, Vector{Float64}};
        mindat=nothing, Kmax=200, shape=0.001, scale=0.001, verbose=false)
    dim = size(X, 2)
    prior = BLMHyper(dim, shape, scale)
    return auto_partition_blm(X, y, bounds, prior; mindat=mindat, Kmax=Kmax,
        verbose=verbose)
end


function predict(mod::PartitionModel, X::Matrix{Float64})
    P = get_P(mod)
    K = length(P)
    N = size(X, 1)
    predfull = Vector{Float64}(undef, N)
    Xvec, rows = partition(X, P; track=true)
    for k in 1:K
        m = get_lm(mod, k)
        preds = predict(m, Xvec[k])
        for i in 1:length(preds)
            predfull[rows[k][i]] = preds[i]
        end
    end
    return predfull
end


function predfun(mod::PartitionModel)
    funs = predfun.(get_modvec(mod))
    P = get_P(mod)
    upper = get_upper(P)
    function f(x::Vector{Float64})
        k = which_subset(x, P, upper)
        return funs[k](x)
    end
end


# Implement non-sequential updates later.
# function fit!(mod::PartitionModel, X::Matrix{Float64}, y::Vector{Float64})
#     P = get_P(mod)
#     Xvec, yvec = partition(X, P, y)
#     parent_scale = get_scalepost(mod)
#     for k in 1:length(P)
#         if length(yvec[k]) == 0 continue end
#         lm = get_lm(mod, k)
#         fit!(lm, Xvec[k], yvec[k])
#     end
# end


function fit!(mod::PartitionModel, x::Vector{Float64}, y::Float64)
    if !isnothing(get_logev(mod))
        set_logev!(mod, nothing)
        set_logdetprior!(mod, nothing)
        set_logdetpost!(mod, nothing)
    end
    P = get_P(mod)
    k = which_subset(x, P)
    lm = get_lm(mod, k)
    scale0 = get_scalepost(mod) - (get_loc_scalepost(mod, k) - get_scaleprior(mod))
    fit!(lm, x, y)
    scale1 = scale0 + (get_loc_scalepost(mod, k) - get_scaleprior(mod))
    shape1 = get_shapepost(mod) + 0.5
    set_scalepost!(mod, scale1)
    set_shapepost!(mod, shape1)
end
