# import LinearAlgebra

using LinearAlgebra: BlasInt, Cholesky

## Custom stuff

"""
    λ, C = posdefλ(Q0, Q1)

Provide an initial `λ` such that `Q0 + λ*Q1` is positive-semidefinite (or preferably,
positive-definite).  `λ` may be larger than strictly necessary. `C` is the Cholesky
decomposition of `Q0 + λ*Q1`.
"""
function posdefλ(Q0::SymTridiagonal{T}, Q1::Union{Diagonal{T},UniformScaling};
                 growauto=T(1.0001), growfail=T(1.1)) where T
    λ = zero(T)
    C = cholesky(Q0; check=false)
    if issuccess(C)
        return λ, C
    end
    dv, ev = Q0.dv, Q0.ev
    n = length(dv)
    α = 4*(cos(π/(n+1)))^2
    for i = 1:n-1
        # A symmetric tridiagonal matrix M is positive-definite if
        #     α * M[i,i+1]^2 < M[i,i]*M[i+1,i+1]   ∀ i
        # See Johnson, Neumann, and Tsatsomeros, Conditions for the positivity of
        # determinants, Linear and Multilinear Algebra 40: 241-248 (1996).
        # This criterion is sufficient but not necessary.

        # In this case M[i,i+1] = Q0[i,i+1] and M[i,i] = Q0[i,i] + λ*Q1[i,i].
        # Consequently we can turn this into a quadratic inequality in λ
        #       a*λ^2 + b*λ + c > 0
        c = dv[i]*dv[i+1] - α*ev[i]^2
        if c < 0 || dv[i] < 0
            # This term needs some positive λ
            a = T(Q1[i,i]*Q1[i+1,i+1])
            @assert a >= 0   # Q1 must be possemidef
            b = Q1[i,i]*dv[i+1] + Q1[i+1,i+1]*dv[i]
            s = sqrt(b^2 - 4*a*c)
            # Since the quadratic is convex and below 0 at λ=0, only one root has λ > 0.
            # So we always choose the (-b + s)/(2*a) solution. But write it in a
            # numerically-stable way.
            λ0 = b > 0  ?  -2*c/(s + b)  :  (-b + s)/(2*a)
            λ = max(λ, λ0)
        end
    end
    # Due to roundoff we need to check whether this actually produces a posdef matrix
    λ *= T(growauto)
    C = cholesky(Q0 + λ*Q1; check=false)
    while !issuccess(C)
        λ *= T(growfail)
        C = cholesky(Q0 + λ*Q1; check=false)
    end
    return λ, C
end

function posdefλ(Q0::AbstractMatrix{T}, Q1::Union{AbstractMatrix{T},UniformScaling};
                 growauto=T(1.0001), growfail=T(1.1)) where T
    λ = zero(T)
    C = cholesky(Q0; check=false)
    if issuccess(C)
        return λ, C
    end
    if isa(Q1, UniformScaling)
        D, _ = eigs(Q0; nev=1, which=:SR, ritzvec=false)
        λ = -T(D[1]/T(Q1.λ))
    else
        D, _ = eigs(Q0, Q1; nev=1, which=:SR, ritzvec=false)
        λ = -T(D[1])
    end
    @assert λ > 0
    λ *= T(growauto)
    C = cholesky(Q0 + λ*Q1; check=false)
    while !issuccess(C)
        λ *= T(growfail)
        C = cholesky(Q0 + λ*Q1; check=false)
    end
    return λ, C
end

asmatrix(M::UniformScaling{T}, sz) where T = M.λ * eye(T, sz...)
asmatrix(M::AbstractMatrix, sz) = Matrix(M)


## Stuff that should be in stdlib/LinearAlgebra

function cholesky(A::SymTridiagonal{T}, ::Val{false}=Val(false); check::Bool=true) where T
    n = size(A, 1)
    dv = zeros(T, n)
    ev = zeros(T, n-1)
    L = Bidiagonal{T}(dv, ev, :L)
    if A.dv[1] < 0
        check && error("A is not positive-semidefinte")
        dv[1] = A.dv[1]
        info = -1
        return Cholesky(L, 'L', convert(BlasInt, info))
    end
    dv[1] = sqrt(A.dv[1])
    info = 0
    for i = 1:n-1
        ev[i] = A.ev[i]/dv[i]
        Δ = A.dv[i+1] - ev[i]^2
        if Δ < 0
            check && error("A is not positive-semidefinte")
            info = -1
            dv[i+1] = Δ  # return the magnitude of the violation for later inspection
            break
        end
        dv[i+1] = sqrt(Δ)
    end
    return Cholesky(L, 'L', convert(BlasInt, info))
end

# cholesky(A::AbstractMatrix{T}, ::Val{false}=Val(false); check::Bool=true) where T =
#     cholfact(A)
function cholesky(A::StridedMatrix{T}, ::Val{false}=Val(false); check::Bool=true) where T
    L, info = LinearAlgebra.LAPACK.potrf!('L', copy(A))
    check && @assert info == 0
    return Cholesky(L, 'L', info)
end

# issuccess(C::Cholesky) = C.info == 0
function issuccess(C::Cholesky)
    n = size(C, 1)
    F = C.factors
    for i = 1:n
        F[i,i] >= 0 || return false
    end
    return true
end

function Base.:+(S::SymTridiagonal{T}, D::UniformScaling{T}) where T
    dv, ev = copy(S.dv), copy(S.ev)
    n = length(dv)
    for i = 1:n
        dv[i] += D[i,i]
    end
    return SymTridiagonal(dv, ev)
end

function Base.:\(C::Cholesky{T,Bidiagonal{T}}, b::AbstractVector) where T
    if C.factors.isupper #C.factors.uplo == 'U'
        R = C.factors
        return R \ (R' \ b)
    end
    L = C.factors
    return L' \ (L \ b)
end

LinearAlgebra.eigen(A::AbstractMatrix, B::UniformScaling) = eigen(A, asmatrix(B, size(A)))
LinearAlgebra.eigen(A::AbstractMatrix, B::Diagonal) = eigen(A, asmatrix(B, size(A)))
