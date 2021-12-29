
using DelimitedFiles
using Random

include("parzenwindow.jl")
include("gmm.jl")
include("roc.jl")


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

logistic(x) = 1/(1+ℯ^(-x))

function test_model(model, params, valid, test_n, test_a, q)
    pred = zeros(2, 3)
    for (j, data) in zip(1:3, [valid, test_n, test_a])
        for i=1:size(data)[2]
            if logistic(model(params, data[:,i])) > q pred[1, j]+=1
            else pred[2, j]+=1 end
        end
    end
    @show pred[1, 1]/sum(pred[:, 1]), pred[1, 2]/sum(pred[:, 2]), 1-pred[1, 3]/sum(pred[:, 3])
    @show ((pred[1, 2]/sum(pred[:, 2]))*size(test_n)[2] + (pred[2, 3]/sum(pred[:, 3]))*size(test_a)[2])/(size(test_n)[2]+size(test_a)[2])
end


K = 50
ps, gmm_model, gm_model = create_gmm(K, N); # prepare model
EM!(ps, trn_data, K, gmm_model, gm_model, 100); # learn model params
arr = []
for i=1:size(valid_data)[2]
    append!(arr, logistic(gmm_model(ps, valid_data[:, i])))
end
test_model(gmm_model, ps, valid_data, test_data_n, data_anomal, quantile!(arr, 0.1)); # run on valid data
@show roc_auc(gmm_model, ps, hcat(test_data_n, data_anomal), 
        vcat(ones(size(test_data_n)[2], 1), zeros(size(data_anomal)[2], 1)));

