
using DelimitedFiles
using Random
using Printf
using ThreadTools

include("src/parzenwindow.jl")
include("src/gmm.jl")
include("src/roc.jl")


# load data
data_anomal = readdlm("data_anomalyproject/yeast/anomalous.txt")'
data_normal = readdlm("data_anomalyproject/yeast/normal.txt")'
data_anomal = data_anomal[:, shuffle(1:end)]
data_normal = data_normal[:, shuffle(1:end)]

# prepare data
N_normal = size(data_normal)[2]
N_anomal = size(data_anomal)[2]
trn_data = data_normal[:, begin:Int(round(N_normal/2))]
valid_data = data_normal[:, Int(round(N_normal/2)):Int(round(3*N_normal/4))]
test_data_n = data_normal[:, Int(round(3*N_normal/4)):end]
test_data_a = data_anomal[:, begin:Int(round(N_anomal/2))]
N = size(data_normal)[1]


likelihood(model, params, data) = log(mean(tmap(x->model(params, x), eachcol(data))))

function choose_gmm_model(data)
    models_dict = Dict()  # likelihood => [model, params]
    lh::Float64 = 0
    for K=2:10
        ps, gmm_model, gm_model = create_gmm(K, N); # prepare model
        EM!(ps, trn_data, K, gmm_model, gm_model, 60); # learn model params
        lh = likelihood(gmm_model, ps, data)
        models_dict[lh] = [gmm_model, ps]
        @printf("K: %d, likelihood: %.3f\n", K, lh)
    end
    return models_dict[maximum(keys(models_dict))]
end

gmm_model, ps = choose_gmm_model(valid_data)

@printf("Best K: %d\n", size(ps[:μ])[1])

testing_data = hcat(test_data_n, data_anomal);
testing_labels = Vector{Bool}(vcat(ones(size(test_data_n)[2], 1), zeros(size(data_anomal)[2], 1))[:,1]);
eval_report(gmm_model, ps, testing_data, testing_labels);
#precision_recall(gmm_model, ps, testing_data, testing_labels, quantile!(arr, 0.1));