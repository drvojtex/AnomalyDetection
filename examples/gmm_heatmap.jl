
include("../src/gmm.jl")

# 2D Gaussian mixed model on random data with heatmap.

using Plots

K = 3 # cluster numbers
num_samples = 50
ps, gmm_model, gm_model = create_gmm(K, 2) # prepare model

# prepare data
tmp=ones(2, num_samples)
tmp1=ones(2, num_samples)
tmp1[1, :] .= 0
X = hcat(randn(2, num_samples)/4+tmp1*1.6, randn(2, num_samples)/2, randn(2, num_samples)/4+tmp*1.6)

# learn model params
EM!(ps, X, K, gmm_model, gm_model, 100)

# plot results
x = y = LinRange(minimum(X), maximum(X), 100)
z = Float64[gmm_model(ps, [yi, xi]) for xi = x, yi = y]
p = heatmap(x, y, z, colorbar_title="Likelihood", title = "Gaussian mixture model heatmap")
scatter!(p, X[1, :], X[2, :], label="Data", xlabel="x1", ylabel="x2")
savefig(p, "gmm_heatmap.pdf")