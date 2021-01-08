module RecursivePartition

export legendre_next, legendre_poly
include("legendre.jl")

export MVPIndex, index_pcbmat, trunc_pcbmat, mvpindex
include("pcb.jl")

export splitmat, insert_knot!, insert_knot, which_subset, partition
include("partition.jl")

end
