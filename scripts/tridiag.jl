using NPZ
using BenchmarkTools
using Base.Threads
using LoopVectorization

a, b, c, g, r, u = [npzread("data/trimats_$x.npy") for x in ['a', 'b', 'c', 'g', 'r', 'u']]

function tri_solve_vec_col!(N, a, b, c, r, g, u)
    beta = b[1]
    u[1] = r[1] / beta

    @inbounds @fastmath for j in 2:N
        g[j] = c[j-1] / beta
        beta = b[j] - a[j] * g[j]
        u[j] = (r[j] - a[j] * u[j-1]) / beta
    end
    @inbounds @fastmath for k in N-1:-1:1
        u[k] -= g[k+1] * u[k+1]
    end
end

coln(x,i) = view(x,:,i)

function tri_solve_vec!(a, b, c, r, g, u)
    N = size(a, 1)
    Threads.@threads for i in 1:N
        tri_solve_vec_col!(N, coln(a,i), coln(b,i), coln(c,i), coln(r,i), coln(g,i), coln(u,i))
    end
end

@btime tri_solve_vec!(a, b, c, r, g, u)

function is_correct()
    a = [1 2 3; 4 5 6; 7 8 9] .|> Float64
    g = zeros((3,3))
    u = zeros((3,3))
    tri_solve_vec!(a, a .+ 1, a .+ 2, a .+ 3, g, u)
    ≈(u, [-0.04 -0.0537634 -0.05844156; 1.36 1.29032258 1.24675325; 0.06 0.07526882 0.07792208], atol=1e-7)
end
is_correct()
