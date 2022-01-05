
using LinearAlgebra


@doc """
Gaussian kernel function.

# Examples
```jldoctest
y = k(x)
```
where the 'x' is Vector{Float64} sample and 'y' is density in point 'x'.
""" ->
k(x::Vector{Float64}) = √(2*π)*ℯ^(-(x'*x)/2)


@doc """
Function to return Parzen window estimator.

# Examples
```jldoctest
fun = Σ(h, X, f, x)
```
where the 'fun' is parzen window estimator, 'h' is window size,
'X' is set of train samples (size(dim, N) ~ (dimension of data, data count),
'f' is kernel function and 'x' is given Vector{Float64} sample.
""" ->
create_parzen_window(h::Float64, X::Matrix{Float64}, 
                        f, x::Vector{Float64}) = (1/(h*size(X)[1]))*mapreduce(a->f((x-a)/h), +, eachcol(X))
