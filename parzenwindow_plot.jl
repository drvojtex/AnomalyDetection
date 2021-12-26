
using Plots

include("parzenwindow.jl")

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
    model(x) = create_parzen_window(step, data, k, x)
    for i in minimum(data):0.01:maximum(data)
        append!(Y, model(i))
    end
    plot!(minimum(data):0.01:maximum(data), Y, color=colors[Int((step+0.4)*2)], label=string("h = ", string(step)))
end

display(plt)
