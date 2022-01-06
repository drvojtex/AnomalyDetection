
using EvalMetrics
using JSON


@doc """
Function to get get statistical report based on given data.

# Examples
```jldoctest
eval_report(model, Θ, data, targets)
```
where the 'Θ' is dictionary of params, 
'model' is used model, 'data' is set of samples (size(dim, N) ~ (dimension of data, data count) and
targets is a Vector{Bool} of labels to given data.
""" ->
function eval_report(model, θ::Dict{Symbol, Vector}, 
                        data::Matrix{Float64}, targets::Vector{Bool})
    get_probs(model, θ::Dict{Symbol, Vector}, 
                data::Matrix{Float64}) = map(x->model(θ, Array{Float64}(x)), eachcol(data))
    scores = Vector{Float64}(get_probs(model, θ, data))
    JSON.print(binary_eval_report(targets, scores), 4)
end

@doc """
Function to get get statistical report based on given data.

# Examples
```jldoctest
auc = eval_report(model, data, targets, show=false)
```
where the 'model' is used model, 
'data' is set of samples (size(dim, N) ~ (dimension of data, data count),
targets is a Vector{Bool} of labels to given data and shaw is param to print the whole 
report (default false). The function returns auc::Float64 which is area under roc-curve.
""" ->
function eval_report(model, data::Matrix{Float64}, targets::Vector{Bool}, show=false)
    get_probs(model, data::Matrix{Float64}) = map(x->model(Array{Float64}(x)), eachcol(data))
    scores::Vector{Float64} = Vector{Float64}(get_probs(model, data))
    report::Dict{String, Float64} = binary_eval_report(targets, scores)
    if show JSON.print(report, 4) end
    return report["au_roccurve"]
end
