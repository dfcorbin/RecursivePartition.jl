module RecursivePartition

using LinearAlgebra: diagm, logdet, norm
using Distributions: Normal, Uniform
using SpecialFunctions: loggamma
using GLMNet: glmnet
using Random: randperm

export gendat
include("utils.jl")

export legendre_next, legendre_poly
include("legendre.jl")

export MVPIndex, index_pcbmat, trunc_pcbmat, mvpindex
include("pcb.jl")

export splitmat, insert_knot!, insert_knot, which_subset, partition,
    is_contained
include("partition.jl")

export BLMHyper, BayesLinearModel, fit!, predict, predfun, logevidence, PolyBLM
include("regression.jl")

export ModArgs, PartitionHyper, PartitionModel, partition_polyblm, partition_blm,
    auto_partition_blm, auto_partition_polyblm

include("partition_regression.jl")


# Getters
export get_shape, get_scale, get_cov, get_dim, get_scaleprior, get_shapeprior, get_covprior,
    get_covinvprior, get_coeffprior, get_scalepost, get_shapepost, get_covpost,
    get_covinvpost, get_coeffpost, get_N, get_indices, get_loc_X, get_loc_y,
    get_loc_coeffpost, get_P, get_K, get_lm, get_logev, get_modvec, get_logdetprior,
    get_logdetpost, get_prior, get_loc_scalepost

# Setters
export set_scaleprior!, set_covprior!, set_covinvprior!, set_coeffprior!,
    set_scalepost!, set_shapepost!, set_covpost!, set_covinvpost!, set_coeffpost!,
    set_logdetprior!, set_logdetprior!, set_logdetpost!, set_logdetpost!, set_logev!,
    set_lm!, append_modvec!, append_logdetprior!, append_logdetpost!


end
