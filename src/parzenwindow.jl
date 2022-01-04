
k(x) = √(2*π)*ℯ^(-(x'*x)/2)
const Σ(h, X, f, x) = (1/(h*size(X)[1]))*mapreduce(a->f((x-a)/h), +, X)
create_parzen_window(h, X, f, x) = Σ(h, X, f, x)
