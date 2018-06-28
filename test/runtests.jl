using OptimizeQCQP
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

n = 4

# With positive-definite Q0
for i = 1:5
    dv = rand(n) + 1
    ev = -rand(n-1)
    Q0 = SymTridiagonal(dv, ev)
    while !isposdef(Matrix(Q0))
        dv = rand(n) + 1
        ev = -rand(n-1)
        Q0 = SymTridiagonal(dv, ev)
    end
    g0 = randn(n)
    xNewton = Q0 \ (-g0)
    Q1 = 1.0*I
    @test OptimizeQCQP.posdefλ(Q0, Q1) == 0
    cu = 1.1*(xNewton'*Q1*xNewton)/2
    x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu)
    @test (x'*Q1*x)/2 <= 1.01*cu
    @test λ == 0
    cu = 0.3*(xNewton'*Q1*xNewton)/2
    x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu)
    @test (x'*Q1*x)/2 <= 1.01*cu
    @test λ > 0
    Q1 = Diagonal(0.25+rand(n))
    @test OptimizeQCQP.posdefλ(Q0, Q1) == 0
    cu = 1.1*(xNewton'*Q1*xNewton)/2
    x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu)
    @test (x'*Q1*x)/2 <= 1.01*cu
    @test λ == 0
    cu = 0.3*(xNewton'*Q1*xNewton)/2
    x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu)
    @test (x'*Q1*x)/2 <= 1.01*cu
    @test λ > 0
end

for i = 1:5
    dv = randn(n)
    ev = randn(n-1)
    Q0 = SymTridiagonal(dv, ev)
    g0 = randn(n)
    for cu in (1.3, 0.5, 0.1, 0.01, 1e-6)
        Q1 = 1.0*I
        λ = OptimizeQCQP.posdefλ(Q0, Q1)
        checkposdef(Q0 + λ*Q1)
        x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu)
        @test (x'*Q1*x)/2 <= 1.01*cu
        Q1 = Diagonal(0.25+rand(n))
        λ = OptimizeQCQP.posdefλ(Q0, Q1)
        checkposdef(Q0 + λ*Q1)
        x, λ = minimize(Q0, g0, nothing, nothing, Q1, nothing, nothing, cu)
        @test (x'*Q1*x)/2 <= 1.01*cu
    end
end

