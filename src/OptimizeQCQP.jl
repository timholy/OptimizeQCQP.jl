__precompile__(true)

module OptimizeQCQP
using Compat

export minimize, Problem

include("linalg.jl")

function minimize(Q0::AbstractMatrix{T}, g0::AbstractVector{T},
                  ::Nothing, ::Nothing, Q1::Union{AbstractMatrix{T},UniformScaling},
                  ::Nothing, ::Nothing, cu::Real; tol=T(1e-4)) where T
    # Solve the problem:
    #    minimize x'*Q0*x/2 + g0'*x
    #    subject to x'*Q1*x/2 <= cu

    # Pick a safe value of λ that makes Q0 + λ*Q1 positive-definite.
    # Keep track of the smallest known such value in λp
    λp = λ = posdefλ(Q0, Q1)

    ## Construct an initial estimate
    C = cholesky(Q0 + λ*Q1)
    x0 = C \ (-g0)
    xQx = x0'*Q1*x0
    if xQx > 2*cu
        # Since λ only guarantees positivity and does not even try to satisfy the cu
        # constraint, it's possible that this λ is far too small. Try to estimate the right
        # value in terms of a correction Δλ where
        #          x = (C + Δλ*Q1) \ (-g)
        # and then picking Δλ so that x satisfies the constraint.
        # We can expand that inverse in two ways:
        # (1) If C dominates Δλ*Q1, use the expansion inv(C + Δλ*Q1) ≈ inv(C) - Δλ*inv(C)*Q1*inv(C)
        Δx = - (C \ (Q1 * x0))  # coefficient of Δλ in correction to x0
        # Solve for Δλ from (x0 + Δλ*Δx)'*Q1*(x0 + Δλ*Δx) == 2*cu
        a = Δx'*Q1*Δx       # coefficient of Δλ^2
        b = 2*x0'*Q1*Δx     # coefficient of Δλ (guaranteed negative)
        c = xQx - 2*cu      # const (coefficient of Δλ^0)
        s2 = b^2 - 4*a*c
        if s2 >= 0
            ΔλC = 2*c/(-b + sqrt(s2))   # choose the smaller of the two roots
        else
            # it doesn't completely reach the criterion, so choose the minimum
            # (this is likely a sign that solution (2) will be selected)
            ΔλC = -b/(2*a)
        end
        @assert ΔλC >= 0
        xC = x0 + ΔλC*Δx   # this is our estimate of the solution, given this Δλ
        # (2) If Δλ*Q1 dominates C, then just use x ≈ (1/Δλ) * (Q1 \ (-g))  (dropping C altogether)
        x1 = Q1 \ (-g0)                # coefficient of Δλ
        Δλ1 = sqrt(-(g0'*x1)/(2*cu))   # our alternative estimate of Δλ
        x1 ./= Δλ1                     # the alternate estimate of x
        # Pick between them by assessing accuracy in matching
        #     (C + Δλ*Q1) * x = -g0
        errC = Q0*xC .+ (λ + ΔλC).*(Q1*xC) .+ g0
        err1 = Q0*x1 .+ (λ + Δλ1).*(Q1*x1) .+ g0
        absprod(x, y) = abs(x)*abs(y)
        errmagC = sum(absprod(x,y) for (x,y) in zip(errC, xC))
        errmag1 = sum(absprod(x,y) for (x,y) in zip(err1, x1))
        λ = errmagC > errmag1 ? λ + Δλ1 : λ + ΔλC
        C = cholesky(Q0 + λ*Q1)
    end

    ## Newton's method iterative improvement
    # Algorithm 4.3 in Nodecal & Wright, generalized for Q1 != I
    xQxold = T(Inf)
    local x
    iter = 0
    while iter < 10
        λp = min(λ, λp)
        x = C \ (-g0)
        q = Q1*x
        dxdλ = - (C \ q)
        xQx = x'*q
        Δ = cu - xQx/2
        Δλ = Δ/cu * (xQx / (2*(q'*dxdλ)))
        λ = max(λ + Δλ, zero(T))
        abs(xQx - xQxold) < tol*(xQx + xQxold) && break
        xQxold = xQx
        iter += 1
        C = cholesky(Q0 + λ*Q1; check=false)
        iterc = 0
        while !issuccess(C) && iterc < 20
            λ = (λ + λp)/2   # bisect the gap to a known-safe value
            C = cholesky(Q0 + λ*Q1; check=false)
            iterc += 1
        end
        issuccess(C) || error("cholesky factorization failed with $λ (compare $λp)")
    end
    return x, λ
end

# Solve the problem:
#    (approximately) minimize x'*Q0*x/2 + g0'*x
#    subject to cl[i] <= x'*Qi*x/2 + gi'*x <= cu[i]       for i = 1,...,m
#                l[k] <= x[k] <= u[k]                     for k = 1,...,n
#
# Note that equality constraints are handled by setting `cl[i] = cu[i]` and/or
# `l[k] = u[k]`.
#
# For positive-semidefinite Q0 and Qi, the problem is convex, otherwise it is NP-hard
# (hence the "approximate" minimization).

struct Problem{T<:Real,Q0T<:AbstractMatrix{T},g0T<:AbstractVector{T}}
    n::Int
    l::Vector{T}
    u::Vector{T}
    cl::Vector{T}
    cu::Vector{T}
    Q0::Q0T
    g0::g0T
    Qidense::Vector{Matrix{T}}
    Qisparse::Vector{SparseMatrixCSC{T,Int}}
    gidense::Vector{Vector{T}}
    gisparse::Vector{SparseVector{T,Int}}
    issparseQi::Vector{Bool}
    issparsegi::Vector{Bool}
    isposdefQi::Vector{Bool}
end

function Problem(Q0::AbstractMatrix{T}, g0::AbstractVector{T},
                 lower=fill(-T(Inf), length(g0)), upper=fill(T(Inf), length(g0))) where T
    n = length(g0)
    indices(g0) == (Base.OneTo(n),) || throw(DimensionMismatch("g0 indices must start with 1, got $(indices(g0))"))
    indices(Q0) == (Base.OneTo(n), Base.OneTo(n))  || throw(DimensionMismatch("Q0 indices must match g0, got $(indices(g0)) and $(indices(Q0))"))
    indices(lower) == (Base.OneTo(n),) || throw(DimensionMismatch("lower's indices must match g0, got $(indices(lower)) and $(indices(g0))"))
    indices(upper) == (Base.OneTo(n),) || throw(DimensionMismatch("upper's indices must match g0, got $(indices(upper)) and $(indices(g0))"))
    Qidense = Matrix{T}[]
    Qisparse = SparseMatrixCSC{T,Int}[]
    gidense = Vector{T}[]
    gisparse = SparseVector{T,Int}[]
    cl, cu = T[], T[]
    issparseQi = Bool[]
    issparsegi = Bool[]
    isposdefQi = Bool[]
    return Problem{T,typeof(Q0),typeof(g0)}(n, lower, upper, cl, cu,
        Q0, g0, Qidense, Qisparse, gidense, gisparse,
        issparseQi, issparsegi, isposdefQi)
end

iszerom(Q::Matrix)          = false
iszerom(Q::SparseMatrixCSC) = isempty(Q.nzval)

zerom(::Type{T}, n) where T = sparse(Int[], Int[], T[], n, n)
zerov(::Type{T}, n) where T = sparse(Int[], T[], n)

function addconstraint!(P::Problem{T}, Qi::AbstractMatrix, gi::AbstractVector, cl::Real, cu::Real;
                        isposdef::Bool=false) where T
    n = P.n
    indices(gi) == (Base.OneTo(n),) || throw(DimensionMismatch("gi indices must match 1:$n, got $(indices(gi))"))
    indices(Qi) == (Base.OneTo(n), Base.OneTo(n))  || throw(DimensionMismatch("Qi indices must match (1:$n, 1:$n), got $(indices(Qi))"))
    Qsp = issparse(Qi)
    gsp = issparse(gi)
    Qsp ? push!(P.Qisparse, Qi) : push!(P.Qidense, Qi)
    gsp ? push!(P.gisparse, gi) : push!(P.gidense, gi)
    push!(P.cl, cl)
    push!(P.cu, cu)
    push!(P.issparseQi, Qsp)
    push!(P.issparsegi, gsp)
    push!(P.isposdefQi, isposdef)
    return P
end
function addconstraint!(P::Problem{T}, Qi::Real, gi::Real, cl::Real, cu::Real; kwargs...) where T
    @assert Qi == 0 && gi == 0
    n = P.n
    return addconstraint!(P, zerom(T, n), zerov(T, n), cl, cu; kwargs...)
end
function addconstraint!(P::Problem{T}, Qi::Real, gi::AbstractVector, cl::Real, cu::Real; kwargs...) where T
    @assert Qi == 0
    n = P.n
    return addconstraint!(P, zerom(T, n), gi, cl, cu; kwargs...)
end
function addconstraint!(P::Problem{T}, Qi::AbstractMatrix, gi::Real, cl::Real, cu::Real; kwargs...) where T
    @assert gi == 0
    n = P.n
    return addconstraint!(P, Qi, zerov(T, n), cl, cu; kwargs...)
end


function minimize(P::Problem; usefactorization::Bool=!issparse(P.Q0))
    if usefactorization
        return minimize_factorization(P)
    else
        error("not yet implemented")
    end
end

function minimize_factorization(P::Problem{T}, x::AbstractVector, μ) where T
    m, n = length(P.issparseQi), P.n
    indices(x) == (Base.OneTo(n),) || throw(DimensionMismatch("x must have indices 1:$n, got $(linearindices(x))"))
    λx, λc, sx, sc, constrval = initialize_auxvars(P, x)
    error("unfinished")
    # if isempty(λx) && m == 1
    #     # No bounds constraints
end

function initialize_auxvars(P::Problem{T}, x) where T
    m, n = length(P.issparseQi), P.n
    # How many inequality constraints are "real"? (x[k] <= Inf does not count as "real")
    nx = count(isfinite, P.l)  + count(isfinite, P.u)
    nc = count(isfinite, P.cl) + count(isfinite, P.cu)
    # Lagrange multipliers
    λx = zeros(T, nx)
    λc = zeros(T, nc)
    # Slack variables
    sx = zeros(T, nx)
    sc = zeros(T, nc)
    # Temporaries
    constrval = fill(T(NaN), m)
    # Initialize the slacks
    si = 0
    for k = 1:n
        if isfinite(P.l[k])
            sx[si+=1] = min(zero(T), x[k] - P.l[k])
        end
    end
    for k = 1:n
        if isfinite(P.u[k])
            sx[si+=1] = min(zero(T), P.u[k] - x[k])
        end
    end
    si = 0
    for k = 1:m
        if isfinite(P.cl[k])
            sc[si+=1] = min(zero(T), evalconstr!(constrval, P, k, x) - P.cl[k])
        end
    end
    for k = 1:m
        if isfinite(P.cu[k])
            sc[si+=1] = min(zero(T), P.cu[k] - evalconstr!(constrval, P, k, x))
        end
    end
    return λx, λc, sx, sc, constrval
end

end # module
