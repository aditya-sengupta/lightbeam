from juliacall import Main as jl

jl.seval("using Base.Threads")

jl.seval("""
function tri_solve_vec_col_b(N, a, b, x)
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
""")

jl.seval("coln(x,i) = view(x,:,i)")

tri_solve_vec_b = jl.seval("""
    function tri_solve_vec_b(a, b, c, r, g, u)
    N = size(a, 1)
    Threads.@threads for i in 1:N
        tri_solve_vec_col_b(N, coln(a,i), coln(b,i), coln(c,i), coln(r,i), coln(g,i), coln(u,i))
    end
end
""")
