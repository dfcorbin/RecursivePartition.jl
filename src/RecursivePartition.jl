module RecursivePartition

using LinearAlgebra: diagm, logdet, norm
using Distributions: Normal, Uniform
using SpecialFunctions: loggamma
using GLMNet: glmnet

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
    # More funs
export get_shape, get_scale, get_cov, get_dim, get_scaleprior, get_shapeprior, get_covprior,
    get_covinvprior, get_coeffprior, get_scalepost, get_shapepost, get_covpost,
    get_covinvpost, get_coeffpost, get_N, get_indices, set_scaleprior!, set_covprior!,
    set_covinvprior!, set_coeffprior!, set_scalepost!, set_shapepost!, set_covpost!,
    set_covinvpost!, set_coeffpost!
include("regression.jl")

end
