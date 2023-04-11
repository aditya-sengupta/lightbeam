using NPZ
using BenchmarkTools
using Base.Threads
using LoopVectorization

a, b, c, g, r, u = [npzread("data/trimats_$x.npy") for x in ['a', 'b', 'c', 'g', 'r', 'u']]

function tri_solve_vec!(a, b, c, r, g, u)
    """Apply Thomas' method for simultaneously solving a set of tridagonal systems. a, b, c, and r are matrices
    (N rows) where each column corresponds a separate system"""
    
    N = size(a, 1)
    beta = copy(b[1,:])
    u[1,:] = r[1,:] ./ beta
    
    @inbounds @fastmath for j in 2:N
        @inbounds for i in 1:N
            g[j,i] = c[j-1,i] / beta[i]
        end
        @inbounds for i in 1:N
            beta[i] = b[j,i] - a[j,i] * g[j,i]
            u[j,i] = (r[j,i] - a[j,i] * u[j-1,i]) / beta[i]
        end
    end

    @inbounds @fastmath for k in N-1:-1:1
        for i in 1:N
            u[k,i] = u[k,i] - g[k+1,i] * u[k+1,i]
        end
    end
end

function tri_solve_vec_col!(a, b, c, r, g, u)
    N = length(a)
    beta = b[1]
    u[1] = r[1] / beta

    @turbo for j in 2:N
        g[j] = c[j-1] / beta
        beta = b[j] - a[j] * g[j]
        u[j] = (r[j] - a[j] * u[j-1]) / beta
    end
    @turbo for k in N-1:-1:1
        u[k] = u[k] - g[k+1] * u[k+1]
    end
end

function is_correct()
    a = [1 2 3; 4 5 6; 7 8 9] .|> Float64
    g = zeros((3,3))
    u = zeros((3,3))
    tri_solve_vec!(a, a .+ 1, a .+ 2, a .+ 3, g, u)
    ≈(u, [-0.04 -0.0537634 -0.05844156; 1.36 1.29032258 1.24675325; 0.06 0.07526882 0.07792208], atol=1e-7)
end
is_correct()

@btime tri_solve_vec!(a, b, c, r, g, u)

@btime for pars in zip([eachcol(x) for x in [a, b, c, r, g, u]]...)
    tri_solve_vec_col!(pars...)
end
