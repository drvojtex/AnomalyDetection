
using LinearAlgebra, Statistics
using ThreadTools


sample_eachcol_type = SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}

# GMM
@doc """
Function to initialize Gaussian mixture model with 'K' components and input vectors of 'dim' dimension.

# Examples
```jldoctest
Θ, gmm, gm = create_gmm(K, dim) 
```
where the 'Θ' is dictionary of params, 'gmm' is Gaussian mixture model and 'gm' is a single Gaussian model.
""" ->
function create_gmm(K::Int64, dim::Int64)
    μ_coefs::Vector{Vector{Float64}} = [c[:] for c in eachcol(randn(dim, K))]
    Σ_coefs::Vector{Matrix{Float64}} = [randn(dim, dim) for _ in 1:K]
    α_coefs::Vector{Float64} = randn(K)
    coefs::Dict{Symbol, Vector} = Dict(:μ=>μ_coefs, :Σ=>Σ_coefs, :α=>α_coefs)
    
    function gmm(Θ::Dict{Symbol, Vector}, x::Vector{Float64})
        n = size(x)[1]
        return mapreduce(usa->usa[3]*(1/((2*π)^(n/2)*det(usa[2])^(1/2)))*
                            (ℯ^(-((x-usa[1])'*inv(usa[2])*(x-usa[1])/2))), +, 
                                zip(Θ[:μ], Θ[:Σ], Θ[:α]))
    end
    gm(μ::Vector{Float64}, 
        Σ::Matrix{Float64}, 
        x::Vector{Float64}) = (1/((2*π)^(size(x)[1]/2)*det(Σ)^(1/2)))*(ℯ^(-((x-μ)'*inv(Σ)*(x-μ)/2)))
    return coefs, gmm, gm
end


# EM algorithm
@doc """
Function to learn params of Gaussian mixture model with the usage of Expectation Maximization Algorithm.

# Examples
```jldoctest
EM!(Θ, X, K, gmm, gm, steps) 
```
where the 'Θ' is dictionary of params to be learned, 
'X' is set of train data (size(dim, N) ~ (dimension of data, data count), 
'gmm' is Gaussian mixture model, 'gm' is a single Gaussian model and 'steps' is count of algorithm steps.
""" ->
function EM!(Θ::Dict{Symbol, Vector}, X::Matrix{Float64}, K::Int64,
             gmm, gm, steps::Int64)
    # Initialize
    N::Int64 = size(X)[2]
    Θ[:Σ] = fill(cov(X, dims=2), K)
    Θ[:α] = fill(1/K, K)
    Θ[:μ] = [X[:, k] for k::Int64 in 1:K]

    saturate_cov(Σ) = det(Σ) < 10e-10 ? Σ+I*10e-5 : Σ
    Θ[:Σ] = tmap(x->saturate_cov(x), Θ[:Σ])

    E_step(k::Int64, x::Vector{Float64}) = Θ[:α][k]*gm(Θ[:μ][k], Θ[:Σ][k], x)/gmm(Θ, x)

    for _=1:steps      
        # E-step
        γ = mapreduce(permutedims, vcat, [tmap(xᵢ::sample_eachcol_type->
                E_step(k, Vector{Float64}(xᵢ)), eachcol(X)) for k::Int64 in 1:K])'
        # M-step
        Θ[:α] = vec(mean(γ, dims=2))
        Γ = sum(γ, dims=1)[1, :]
        for k::Int64=1:K
            μₖ::Vector{Float64} = mapreduce(ix->γ[ix[1], k]*
                    Vector{Float64}(ix[2]), +, zip(1:N, eachcol(X)))/Γ[k]
            Σₖ::Matrix{Float64} = mapreduce(ix->γ[ix[1], k]*
                    (Vector{Float64}(ix[2])-μₖ)*(Vector{Float64}(ix[2])-μₖ)', +, zip(1:N, eachcol(X)))/Γ[k]
            Θ[:Σ][k] = saturate_cov(Σₖ)
            Θ[:μ][k] = μₖ
        end 
    end
    Θ
end
