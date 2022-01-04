
using Plots
using EvalMetrics

function get_probs(model, θ, data)
    probs = []
    for i=1:size(data)[2]
        append!(probs, model(θ, data[:,i]))
    end
    return probs
end

function eval_report(model, θ, data, targets)
    scores = Vector{Float64}(get_probs(model, θ, data))
    print(binary_eval_report(targets, scores))
end

function precision_recall(model, θ, data, targets, ϵ) 
    scores = Vector{Float64}(get_probs(model, θ, data))
    cm = ConfusionMatrix(targets, scores, ϵ)
    println("precision: ", precision(cm))
    println("recall: ", recall(cm))
end
