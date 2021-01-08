module RecursivePartition

export legendre_next, legendre_poly
include("legendre.jl")

export MVPIndex, index_pcbmat, trunc_pcbmat, mvpindex 
include("pcb.jl")

end
