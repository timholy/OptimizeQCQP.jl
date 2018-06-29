# import LinearAlgebra

using Base.LinAlg: BlasInt, Cholesky

## Custom stuff

"""
    λ = posdefλ(Q0, Q1)

Provide an initial `λ` such that `Q0 + λ*Q1` is positive-semidefinite (or preferably,
positive-definite).  `λ` may be larger than strictly necessary.
"""
function posdefλ(Q0::SymTridiagonal{T}, Q1::Union{Diagonal{T},UniformScaling}) where T
    λ = zero(T)
    if issuccess(cholesky(Q0; check=false))
        return zero(T)
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
            # try
            #     @assert α*ev[i]^2 <= (1+1e-8)*(dv[i] + λ*Q1[i,i])*(dv[i+1] + λ*Q1[i+1,i+1])
            # catch
            #     @show  α*ev[i]^2 (dv[i] + λ*Q1[i,i])*(dv[i+1] + λ*Q1[i+1,i+1])
            #     rethrow()
            # end
        end
    end
    return λ
end

function posdefλ(Q0::AbstractMatrix{T}, Q1::Union{AbstractMatrix{T},UniformScaling}) where T
    isposdef(Q0) && return zero(T)
    if isa(Q1, UniformScaling)
        D, _ = eigs(Q0; nev=1, which=:SR, ritzvec=false)
        λ = -D[1]/T(Q1.λ)
    else
        D, _ = eigs(Q0, Q1; nev=1, which=:SR, ritzvec=false)
        λ = -D[1]
    end
    @assert λ > 0
    λ *= T(1.0001)   # more likely to make it posdef on the first iteration
    while !isposdef(Q0 + λ*Q1)
        λ *= T(1.1)
    end
    return λ
    # D, _ = eig(Q0, Q1)
    # if !issorted(D)
    #     @show Q0 Q1 D
    # end
    # @assert issorted(D)
    # Dmin = D[1]
    # # Theoretically we could just return -Dmin, but due to roundoff error we have
    # # to be more cautious
    # for d in D
    #     λ = -Dmin + (d-Dmin)
    #     isposdef(Q0+λ*Q1) && return λ
    # end
    # error("No suitable λ found, eigenvalues: $D")
end

asmatrix(M::UniformScaling{T}, sz) where T = M.λ * eye(T, sz...)
asmatrix(M::AbstractMatrix, sz) = Matrix(M)


## Stuff that should be in stdlib/LinearAlgebra

function cholesky(A::SymTridiagonal{T}, ::Val{false}=Val(false); check::Bool=true) where T
    n = size(A, 1)
    dv = zeros(T, n)
    ev = zeros(T, n-1)
    L = Bidiagonal{T}(dv, ev, false) #'L')  # {} needed to avoid copying
    if A.dv[1] < 0
        check && error("A is not positive-semidefinte")
        dv[1] = A.dv[1]
        info = -1
        return Cholesky(L, 'L') #, convert(BlasInt, info))
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
    return Cholesky(L, 'L') #, convert(BlasInt, info))
end

# cholesky(A::AbstractMatrix{T}, ::Val{false}=Val(false); check::Bool=true) where T =
#     cholfact(A)
function cholesky(A::StridedMatrix{T}, ::Val{false}=Val(false); check::Bool=true) where T
    L, info = LinAlg.LAPACK.potrf!('L', copy(A))
    check && @assert info == 0
    return Cholesky(L, 'L')
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

Base.eigfact(A::AbstractMatrix, B::UniformScaling) = eigfact(A, asmatrix(B, size(A)))
Base.eigfact(A::AbstractMatrix, B::Diagonal) = eigfact(A, asmatrix(B, size(A)))
