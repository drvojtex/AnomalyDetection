
using DelimitedFiles
using Random

include("parzenwindow.jl")
include("gmm.jl")


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
test_data_a = data_normal[:, begin:Int(round(N_anomal/2))]
N = size(data_normal)[1]

function test_model(model, params, data)
    arr = []
    for i=1:size(data)[2]
        append!(arr, model(params, data[:,i]))
    end
    pred = zeros(2)
    for i=1:size(data)[2]
        if model(params, data[:,i]) > quantile!(arr, 0.1) pred[1]+=1
        else pred[2]+=1 end
    end
    acc = pred[1]/sum(pred)
    return acc
end

# Select hyperparameter by accuracy on valid data
acc_valid = []
for K=2:20
    ps, gmm_model, gm_model = create_gmm(K, N) # prepare model
    EM!(ps, trn_data, K, gmm_model, gm_model, 30) # learn model params
    acc = test_model(gmm_model, ps, valid_data) # run on valid data
    append!(acc_valid, acc)
    @show K, acc
end
best_K = argmax(acc_valid)+1
@show best_K

