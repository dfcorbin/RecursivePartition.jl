using RecursivePartition
using Test

@testset "Legendre Polynomials" begin
    include("test_legendre.jl")
end

@testset "Polynomial Chaos" begin
    include("test_pcb.jl")
end

@testset "Partition" begin
    include("test_partition.jl")
end
