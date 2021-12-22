
using Plots

# Parzen window
k(x) = √(2*π)*ℯ^(-(x'*x)/2)
const Σ(h, X, f, x) = (1/(h*size(X)[1]))*mapreduce(a->f((x-a)/h), +, X)

# Data generation
data1 = ones(20)*2+randn(20)
data2 = ones(10)*6+randn(10)
data = vcat(data1, data2)

# Plot data
plt = plot()
scatter!(data1, zeros(20), color="black", label = "")
scatter!(data2, zeros(10), color="black", label = "")

# Plot parzen window function
colors = ["red", "green", "blue", "black"]
for step in 0.1:0.5:1.6
    Y = []
    for i in minimum(data):0.01:maximum(data)
        append!(Y, Σ(step, data, k, i))
    end
    plot!(minimum(data):0.01:maximum(data), Y, color=colors[Int((step+0.4)*2)], label=string("h = ", string(step)))
end

display(plt)
