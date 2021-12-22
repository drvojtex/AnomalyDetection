
include("gmm.jl")

# load data
data_anomal = readdlm("data_anomalyproject/pendigits/anomalous.txt")'
data_normal = readdlm("data_anomalyproject/pendigits/normal.txt")'

# creat model
K = 2 # cluster numbers
N = size(data_normal)[2]
data_normal[:, Int(round(N/2))]
ps, gmm_model, gm_model = create_gmm(K, Int(round(N/2))) # prepare model

# learn model params
EM!(ps, data_normal, K, gmm_model, gm_model, 10)

# test model on train data
pred = zeros(2)
for i=1:Int(round(N/2))
    if i%100==0 @show i end
    if gmm_model(ps, data_normal[:,i]) > 0.01 pred[1]+=1
    else pred[2]+=1 end
end

println(pred[1]/sum(pred))
