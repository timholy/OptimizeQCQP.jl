using OptimizeQCQP
using JuMP, Ipopt   # for comparison purposes
using Base.Test

function checkposdef(M::SymTridiagonal)
    dv, ev = M.dv, M.ev
    n = length(dv)
    α = 4*(cos(π/(n+1)))^2
    for i = 1:n-1
        @test α*ev[i]^2 <= (1+1e-8)*dv[i]*dv[i+1]
    end
    return nothing
end
function checkposdef(M)
    @assert isposdef(M)
    return nothing
end

function checksolution(Q0, g0, Q1, cu; print_level=0)
    solver = IpoptSolver(print_level=print_level,
                         sb="yes")
                         #  tol=tol,
                         #  max_iter=max_iter)
    model = Model(solver=solver)
    n = length(g0)
    @variable(model, x[1:n])
    @constraint(model, x'*Q1*x <= 2*cu)
    @objective(model, Min, (x'*Q0*x)/2 + g0'*x)
    status = solve(model)
    status == :Optimal || error("checksolution solution not optimal")
    y = getvalue(x)
    # Ipopt sometimes has relatively low accuracy, so make sure we enforce the constraint
    ymag = (y'*Q1*y)/2
    if ymag > cu
       y .*= sqrt(cu/ymag)
    end
    return y, (y'*Q0*y)/2 + g0'*y
end

function symtridiag(n, isposdef)
    if isposdef
        dv = rand(n) + 2
        ev = -rand(n-1)
        return SymTridiagonal(dv, ev)
    else
        dv = randn(n)
        ev = randn(n-1)
        return SymTridiagonal(dv, ev)
    end
end

function symdense(eigvals)
    n = length(eigvals)
    U, _ = qr(randn(n, n))
    Q = U*Diagonal(eigvals)*U'
    return (Q + Q')/2
end

@testset "posdefλ" begin
    A = symdense([-1.4, 0.2, 3])
    λ = OptimizeQCQP.posdefλ(A, I)
    @test λ >= 1.4 && λ < 2
    λ = OptimizeQCQP.posdefλ(A, 2*I)
    @test λ >= 0.7 && λ < 1
    λ = OptimizeQCQP.posdefλ(A, 0.5*I)
    @test λ >= 2.8 && λ < 4
    B = Diagonal(fill(0.5, 3))
    λ = OptimizeQCQP.posdefλ(A, B)
    @test λ >= 2.8 && λ < 4
    λ = OptimizeQCQP.posdefλ(A, Matrix(B))
    @test λ >= 2.8 && λ < 4

    A = SymTridiagonal([-1, 0.1, 2], [0.0, 0.0])
    λ = OptimizeQCQP.posdefλ(A, I)
    @test λ ≈ 1
    λ = OptimizeQCQP.posdefλ(A, 2*I)
    @test λ ≈ 0.5
    A = SymTridiagonal([-1, 0.1, 2], [1.0, 1.0])
    λ = OptimizeQCQP.posdefλ(A, I)
    @test λ >= -minimum(eig(Matrix(A))[1])
end

@testset "Q0/g0/Q1" begin
    n = 4
    # With positive-definite Q0
    for gen in (()->symdense([0.1, 1, 2, 3]),
                ()->symtridiag(4, true),)
        for i = 1:5
            Q0 = gen()
            g0 = randn(n)
            xNewton = Q0 \ (-g0)
            Q1 = 1.0*I
            @test OptimizeQCQP.posdefλ(Q0, Q1) == 0
            cu = 1.1*(xNewton'*Q1*xNewton)/2
            x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu; tol=1e-8)
            @test (x'*Q1*x)/2 <= 1.01*cu
            @test λ == 0
            x_ipopt, val_ipopt = checksolution(Q0, g0, Q1, cu)
            @test (x'*Q0*x)/2 + g0'*x <= val_ipopt + 1e-5*abs(val_ipopt)
            cu = 0.3*(xNewton'*Q1*xNewton)/2
            x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu; tol=1e-8)
            @test (x'*Q1*x)/2 <= 1.01*cu
            @test λ > 0
            x_ipopt, val_ipopt = checksolution(Q0, g0, Q1, cu)
            @test (x'*Q0*x)/2 + g0'*x <= val_ipopt + 1e-5*abs(val_ipopt)
            Q1 = Diagonal(0.25+rand(n))
            @test OptimizeQCQP.posdefλ(Q0, Q1) == 0
            cu = 1.1*(xNewton'*Q1*xNewton)/2
            x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu; tol=1e-8)
            @test (x'*Q1*x)/2 <= 1.01*cu
            @test λ == 0
            x_ipopt, val_ipopt = checksolution(Q0, g0, Q1, cu)
            @test (x'*Q0*x)/2 + g0'*x <= val_ipopt + 1e-5*abs(val_ipopt)
            cu = 0.3*(xNewton'*Q1*xNewton)/2
            x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu; tol=1e-8)
            @test (x'*Q1*x)/2 <= 1.01*cu
            @test λ > 0
            x_ipopt, val_ipopt = checksolution(Q0, g0, Q1, cu)
            @test (x'*Q0*x)/2 + g0'*x <= val_ipopt + 1e-5*abs(val_ipopt)
        end
    end
    # With indefinite Q0
    for gen in (()->symdense([-1, -0.1, 2, 3]),
                ()->symtridiag(4, false),)
        for i = 1:5
            Q0 = gen()
            g0 = randn(n)
            for cu in (1.3, 0.5, 0.1, 0.01, 1e-6)
                Q1 = 1.0*I
                λ = OptimizeQCQP.posdefλ(Q0, Q1)
                checkposdef(Q0 + λ*Q1)
                x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu; tol=1e-8)
                @test (x'*Q1*x)/2 <= 1.01*cu
                x_ipopt, val_ipopt = checksolution(Q0, g0, Q1, cu)
                @test (x'*Q0*x)/2 + g0'*x <= val_ipopt + 1e-5*abs(val_ipopt)
                Q1 = Diagonal(0.25+rand(n))
                λ = OptimizeQCQP.posdefλ(Q0, Q1)
                checkposdef(Q0 + λ*Q1)
                x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu; tol=1e-8)
                @test (x'*Q1*x)/2 <= 1.01*cu
                x_ipopt, val_ipopt = checksolution(Q0, g0, Q1, cu)
                @test (x'*Q0*x)/2 + g0'*x <= val_ipopt + 1e-5*abs(val_ipopt)
            end
        end
    end
end
