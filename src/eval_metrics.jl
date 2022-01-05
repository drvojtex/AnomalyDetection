
using EvalMetrics
using JSON


@doc """
Function to get scores of models for given samples.

# Examples
```jldoctest
scores = (get_probs(model, θ, data))
```
where the 'Θ' is dictionary of params, 'model' is used model and 
'data' is set of samples (size(dim, N) ~ (dimension of data, data count).
Scores is Vector{Float64}.
""" ->
get_probs(model, θ::Dict{Symbol, Vector}, 
            data::Matrix{Float64}) = map(x->model(θ, x), eachcol(data))


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
    scores = Vector{Float64}(get_probs(model, θ, data))
    JSON.print(binary_eval_report(targets, scores), 4)
end
