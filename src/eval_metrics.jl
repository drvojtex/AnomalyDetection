
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
                        data::Matrix{Float64}, targets::Vector{Bool}, show=false)
    get_probs(model, θ::Dict{Symbol, Vector}, 
                data::Matrix{Float64}) = tmap(x->model(θ, Array{Float64}(x)), eachcol(data))
    scores = Vector{Float64}(get_probs(model, θ, data))
    report::Dict{String, Float64} = binary_eval_report(targets, scores)
    if show JSON.print(report, 4) end
    return report["au_roccurve"]
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
    get_probs(model, data::Matrix{Float64}) = tmap(x->model(Array{Float64}(x)), eachcol(data))
    scores::Vector{Float64} = Vector{Float64}(get_probs(model, data))
    report::Dict{String, Float64} = binary_eval_report(targets, scores)
    if show JSON.print(report, 4) end
    return report["au_roccurve"]
end


@doc """
    Wilcoxon sing-rank non-parametric test to check if two populations have the 
    same median. 

    # Examples
    ```jldoctest
    W::Int64, z::Float64, b::Bool = wilcoxon(X::Vector{Float64}, Y::Vector{Float64})
    ```
    where 'W' is W-value, 'z' is z-score and 'b' is the test result for the First type error 0.05. 
    When 'b' is true, the populations 'X' and 'Y' have got the same median.
    """ ->
function wilcoxon(X::Vector{Float64}, Y::Vector{Float64})
    xydiff::Vector{Float64} = filter(x->x!=0, X.-Y) 
    df = DataFrame(diff = xydiff, absdiff = abs.(xydiff))
    sort!(df, [order(:absdiff)])
    df[!, :sgn] = sign.(df[!, :diff])
    df[!, :Rᵢ] = 1:length(df[!, :diff])
    W::Int64 = min(sum(filter(x->x.sgn==-1, df)[!, :Rᵢ]), 
                    sum(filter(x->x.sgn==1, df)[!, :Rᵢ]))
    z::Float64 = mapreduce(xy->xy[1]*xy[2], +, zip(df[!,:sgn], df[!,:Rᵢ]))
    N::Int64 = size(df)[1]
    z = z/sqrt(N*(N+1)*(2*N+1)/6)
    return W, z, abs(z) <= 1.96
end
