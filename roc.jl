
using Plots
using EvalMetrics

function get_probs(model, θ, data)
    probs = []
    for i=1:size(data)[2]
        append!(probs, model(θ, data[:,i]))
    end
    return probs
end

function roc_auc(model, θ, data, targets)
    scores = get_probs(model, θ, data)
    auc = auc_trapezoidal(prcurve(targets, scores)...)
    println("auc: ", auc)
end

function recall(model, θ, data, y, ϵ) 
    ŷ = get_probs(model, θ, data)
    fn = sum(map(x->x[1] == 0 && x[2] == 1, zip(map(x->x>ϵ, ŷ), y)))
    tp = sum(map(x->x[1] == 1 && x[2] == 1, zip(map(x->x>ϵ, ŷ), y)))
    tpr = tp/(fn+tp)
    println("recall: ", tpr)
end

function precision(model, θ, data, y, ϵ) 
    ŷ = get_probs(model, θ, data)
    fp = sum(map(x->x[1] == 1 && x[2] == 0, zip(map(x->x>ϵ, ŷ), y)))
    tp = sum(map(x->x[1] == 1 && x[2] == 1, zip(map(x->x>ϵ, ŷ), y)))
    tpr = tp/(fp+tp)
    println("precision: ", tpr)
end
