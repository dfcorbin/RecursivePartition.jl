using RecursivePartition
using Test, Random

function approxeq(v1::Float64, v2::Float64, tol::Float64)
    if tol < 0
        throw(ArgumentError("tol must be ≥ 0"))
    end
    if abs(v1 - v2) < tol
        return true
    end
    return false
end


function approxeq(v1::Vector{Float64}, v2::Vector{Float64}, tol::Float64)
    for i = 1:length(v1)
        if !approxeq(v1[i], v2[i], tol)
            return false
        end
    end
    return true
end

@testset "Legendre Polynomials" begin
    include("test_legendre.jl")
end

@testset "Polynomial Chaos" begin
    include("test_pcb.jl")
end

@testset "Partition" begin
    include("test_partition.jl")
end

@testset "Regression" begin
    include("test_regression.jl")
end

@testset "partition-regression" begin
    include("test_partition_regression.jl")
end
