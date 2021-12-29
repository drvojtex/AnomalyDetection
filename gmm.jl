
using LinearAlgebra, Statistics
using ThreadTools


# GMM
function create_gmm(K::Int64, dim::Int64)
    μ_coefs = [c[:] for c in eachcol(randn(dim, K))]
    Σ_coefs = [randn(dim, dim) for _ in 1:K]
    α_coefs = randn(K)
    coefs = Dict(:μ=>μ_coefs, :Σ=>Σ_coefs, :α=>α_coefs)
    function gmm(Θ, x)
        n = size(x)[1]
        return mapreduce(usa->usa[3]*(1/((2*π)^(n/2)*det(usa[2])^(1/2)))*
                            (ℯ^(-((x-usa[1])'*inv(usa[2])*(x-usa[1])/2))), +, 
                                zip(Θ[:μ], Θ[:Σ], Θ[:α]))
    end
    gm(μ, Σ, x) = (1/((2*π)^(size(x)[1]/2)*det(Σ)^(1/2)))*(ℯ^(-((x-μ)'*inv(Σ)*(x-μ)/2)))
    return coefs, gmm, gm
end


# EM algorithm
function EM!(Θ, X, K, gmm, gm, steps)
    # Initialize
    N = size(X)[2]
    Θ[:Σ] = fill(cov(X, dims=2), K)
    Θ[:α] = fill(1/K, K)
    Θ[:μ] = [X[:, k] for k in 1:K]

    E_step(k, x) = Θ[:α][k]*gm(Θ[:μ][k], Θ[:Σ][k], x)/gmm(Θ, x)

    for step=1:steps      
        #@show step
        
        # E-step
        γ = mapreduce(permutedims, vcat, [tmap(xᵢ->E_step(k, xᵢ), eachcol(X)) for k in 1:K])'

        # M-step
        Θ[:α] = vec(mean(γ, dims=2))
        Γ = sum(γ, dims=1)[1, :]
        for k=1:K
            μₖ = mapreduce(ix->γ[ix[1], k]*ix[2], +, zip(1:N, eachcol(X)))/Γ[k]
            Σₖ = mapreduce(ix->γ[ix[1], k]*(ix[2]-μₖ)*(ix[2]-μₖ)', +, zip(1:N, eachcol(X)))/Γ[k]
            if det(Σₖ) < 10e-18 Σₖ+=I*10e-10 end
            Θ[:μ][k] = μₖ
            Θ[:Σ][k] = Σₖ
        end 
    end
    Θ
end
