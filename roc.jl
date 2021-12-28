
using NumericalIntegration
using Plots

function get_probs(model, θ, data)
    probs = []
    for i=1:size(data)[2]
        append!(probs, model(θ, data[:,i]))
    end
    return probs
end

function fpr_tpr(ŷ, y, ϵ)
    fp = sum(map(x->x[1] == 1 && x[2] == 0, zip(map(x->x>ϵ, ŷ), y)))
    tn = sum(map(x->x[1] == 0 && x[2] == 0, zip(map(x->x>ϵ, ŷ), y)))
    fpr = fp/(fp+tn)

    fn = sum(map(x->x[1] == 0 && x[2] == 1, zip(map(x->x>ϵ, ŷ), y)))
    tp = sum(map(x->x[1] == 1 && x[2] == 1, zip(map(x->x>ϵ, ŷ), y)))
    tpr = tp/(fn+tp)

    return fpr, tpr
end

function roc_auc(model, θ, data, labels)
    probs = get_probs(model, θ, data)
    fpr_tpr_arr = []
    for ϵ=0:((maximum(probs)-minimum(probs))/(size(data)[2]*maximum(probs))):1
        fpr, tpr = fpr_tpr(probs, labels, ϵ)
        append!(fpr_tpr_arr, [(fpr, tpr)])
    end
    sort!(fpr_tpr_arr)
    
    # trapezoid integration
    auc = integrate(map(x->x[1], fpr_tpr_arr), 
                    map(x->x[2], fpr_tpr_arr))
    return auc
end

function rocplot(model, θ, data, labels)
    probs = get_probs(model, θ, data)
    fpr_tpr_arr = []
    for ϵ=0:((maximum(probs)-minimum(probs))/size(data)[2]):1
        fpr, tpr = fpr_tpr(probs, labels, ϵ)
        append!(fpr_tpr_arr, [(fpr, tpr)])
    end
    sort!(fpr_tpr_arr)
    plot(map(x->x[1], fpr_tpr_arr), map(x->x[2], fpr_tpr_arr))
end
