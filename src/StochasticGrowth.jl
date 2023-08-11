using LinearAlgebra, Statistics
using LaTeXStrings, Plots, Interpolations, Roots, Random, Parameters
using Optim

# ========== Method 1: value function iteration ==========

# a bellman operator
function T_VFI(v0; params, c_lb=1e-10)
    (; y_grid, ξ, β, u, f) = params

    v1 = similar(v0)    # new value function
    c  = similar(v0)    # optimal policy function
    v0_func = LinearInterpolation(y_grid, v0, extrapolation_bc=Line())

    for (i, y) in enumerate(y_grid)
        res = maximize(c -> -(u(c) + β * mean(v0_func.(f(y - c) .* ξ))), c_lb, y)
        v1[i] = maximum(res)
        c[i]  = maximizer(res)
    end

    return (;v=v1, c)
end


# ========== Method 2: value function iteration ==========

# a Coleman operator
function K_PFI(σ0; params, c_lb=1e-10)
    (; y_grid, ξ, β, u, du, f, df) = params

    σ1 = similar(σ0)    # new policy function
    σ0_func = LinearInterpolation(y_grid, σ0, extrapolation_bc=Line())

    for (i, y) in enumerate(y_grid)
        fn2solve(c) = du(c) - mean(β .* du.(σ0_func.(f(y - c) .* ξ)) .* df(y - c) .* ξ)
        σ1[i] = find_zero(fn2solve, (c_lb, y-c_lb))
    end

    return (;σ=σ1)
end


# ========== Method 3: endogenous grid method ==========

# a Coleman operator
function K_EGM(σ0; params, c_lb=1e-10)
    (; k_grid, ξ, β, u, du, u_inv, f, df) = params

    c_grid = similar(k_grid)

    for (i, k) in enumerate(k_grid)
        val = mean(β .* du.(σ0.(f(k) .* ξ)) .* df(k) .* ξ)
        c_grid[i] = u_inv(val)
    end

    y_grid = k_grid + c_grid
    σ1 = LinearInterpolation(y_grid, c_grid, extrapolation=Line())

    return (;σ=σ1)
end


function VFI(v0; params, MaxIter=100, tol=1e-8)

    # plot initial value function
    plt1 = plot(title="VFI")
    y_grid = params.y_grid
    plot!(plt1, y_grid, v0, color = :black, label="v0")

    plt2 = plot(title="Policy functions in VFI")   # plot policy function

    for i in 1:MaxIter
        print(v0)
        res = T_VFI(v0; params)
        v1, c1 = res.v, res.c
        plot!(plt1, y_grid, v1, color = RGBA(i/MaxIter, 0, 1-i/MaxIter, 0.8))
        plot!(plt2, y_grid, c1, color = RGBA(i/MaxIter, 0, 1-i/MaxIter, 0.8))
        v0 = v1
        print(v0)

        if maximum(abs.(v1 - v0)) <= tol
            println("Converged in $i iterations")

            # plot
            display(plt1)
            display(plt2)
            return v0
        end
    end
end


function PFI(σ0; params, MaxIter=100, tol=1e-8)

    # plot initial policy function
    plt = plot(title="PFI")
    y_grid = params.y_grid
    plot!(plt, y_grid, σ0, color = :black, label="σ0")
    
    for i in 1:MaxIter
        println("iter = $i")
        σ1 = K_PFI(σ0; params).σ
        plot!(plt, y_grid, σ1, color = RGBA(i/MaxIter, 0, 1-i/MaxIter, 0.8))
        σ0 = σ1

        if maximum(abs.(σ1 - σ0)) <= tol
            println("Converged in $i iterations")
            display(plt)
            return σ0
        end
    end

end

function PFI_EGM(σ0; params, MaxIter=100, tol=1e-8)

    # plot initial policy function
    plt = plot()
    y_grid = params.y_grid      # retrieve an exogenous y_grid to evaluate PF
    σ0_grid = σ0.(y_grid)
    plot!(plt, y_grid, σ0_grid , color = :black, label="σ0")
    
    for i in 1:MaxIter
        σ1 = K_EGM(σ0; params).σ        # this is a function
        σ1_grid = σ1.(y_grid)
        plot!(plt, y_grid, σ1_grid, color = RGBA(i/MaxIter, 0, 1-i/MaxIter, 0.8))
        σ0 = σ1

        if maximum(abs.(σ1_grid  - σ0_grid )) <= tol
            println("Converged in $i iterations")
            return σ0
        end
    end

end



Params = @with_kw (
    α = 0.65,                   # productivity parameter
    β = 0.95,                   # discount factor
    γ = 1.0,                    # risk aversion
    μ = 0.0,                    # lognorm(μ, σ)
    s = 0.1,                    # lognorm(μ, σ)
    grid_min    = 1e-6,            # smallest grid point
    grid_max    = 4.0,             # largest grid point
    grid_size   = 200,             # grid size
    shock_size  = 1000,          # num of Monte Carlo draws
    u   = (γ == 1 ? log : c->(c^(1-γ)-1)/(1-γ)),        # utility function
    du  = c-> c^(-γ),            # u'
    f   = k -> k^α,                # production function
    df  = k -> α*k^(α-1),        # f'
    y_grid = range(grid_min, grid_max, length = grid_size),
    k_grid = y_grid,
    ξ      = exp.(μ .+ s * randn(shock_size)))


params = Params()

# VFI(zeros(params.grid_size); params)

PFI(params.y_grid; params)