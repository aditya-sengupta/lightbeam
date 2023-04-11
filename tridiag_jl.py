from julia import Main

tri_solve_vecjl = Main.eval("""function tri_solve_vec(a, b, c, r, g, u)
    N = size(a, 1)
    beta = b[1]
    u[1] = r[1]/beta
    
    for j in 2:N
        g[j] = c[j-1]/beta
        beta = b[j] - a[j]*g[j]
        u[j] = (r[j] - a[j]*u[j-1])/beta
    end

    for j in 1:N-2
        k = N-1-j
        u[k] = u[k] - g[k+1]*u[k+1]
    end
end""")

