
using DelimitedFiles
using Random
using Printf
using ThreadTools

include("src/parzenwindow.jl")
include("src/gmm.jl")
include("src/eval_metrics.jl")


# Load all data.
data_anomal = readdlm("data_anomalyproject/yeast/anomalous.txt")'
data_normal = readdlm("data_anomalyproject/yeast/normal.txt")'
data_anomal = data_anomal[:, shuffle(1:end)]
data_normal = data_normal[:, shuffle(1:end)]

# Prepare all data.
N_normal = size(data_normal)[2]
N_anomal = size(data_anomal)[2]
trn_data = data_normal[:, begin:Int(round(N_normal/2))]
valid_data = data_normal[:, Int(round(N_normal/2)):Int(round(3*N_normal/4))]
test_data_n = data_normal[:, Int(round(3*N_normal/4)):end]
test_data_a = data_anomal[:, begin:Int(round(N_anomal/2))]
N = size(data_normal)[1]

# Define function to get likelihood proxy param.
likelihood(model, params, data) = log(mean(tmap(x->model(params, x), eachcol(data))))


@doc """
Function to Choose optimal count of components (hyperparametr) of Gaussian mixture model. The choose is 
performed base on maximization of likelihood (proxy parametr) on given (validation) data. 
Minimum of components is 2, maximum is 10.

# Examples
```jldoctest
model, Θ = choose_gmm_model(data)
```
where the 'Θ' is dictionary of params of the GMM, 
'data' is set of (validation) data (size(dim, N) ~ (dimension of data, data count), 
'model' is a Gaussian mixture model.
""" ->
function choose_gmm_model(data::Matrix{Float64})
    models_dict = Dict{Float64, Vector{Any}}()  # likelihood => [model, params]
    lh::Float64 = 0
    for K=2:10
        ps::Dict{Symbol, Vector}, gmm_model, gm_model = create_gmm(K, N); # prepare model
        EM!(ps, trn_data, K, gmm_model, gm_model, 60); # learn model params
        lh = likelihood(gmm_model, ps, data)
        models_dict[lh] = [gmm_model, ps]
    end
    return models_dict[maximum(keys(models_dict))]
end

# Learn GMM on train data and choose the best count of components by validation data.
gmm_model, ps = choose_gmm_model(valid_data)
@printf("Best K: %d\n", size(ps[:μ])[1])

# Prepare testing data.
testing_data = hcat(test_data_n, data_anomal);
testing_labels = Vector{Bool}(vcat(ones(size(test_data_n)[2], 1), zeros(size(data_anomal)[2], 1))[:,1]);
# Print evaluation report.
eval_report(gmm_model, ps, testing_data, testing_labels);
