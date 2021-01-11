const Matvec = Vector{Matrix{T}} where {T}
const Vecvec = Vector{Vector{T}} where {T}


function gendat(N, SD, f, d; bounds = [-1, 1])
    X = rand(Uniform(bounds[1], bounds[2]), N, d)
    y = reshape(mapslices(f, X; dims = 2), (:)) .+ rand(Normal(0.0, SD), N)
    return (X, y)
end


function woodbury_inv(Ainv::Matrix{Float64}, u::Vector{Float64}, v::Vector{Float64})
    numer::Matrix{Float64} = Ainv * u * v' * Ainv
    denom::Float64 = 1 + v' * Ainv * u
    return Ainv - numer * (1 / denom)
end
