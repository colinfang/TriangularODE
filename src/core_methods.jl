function lagrange_polynomials(x::Float64, x0::Float64, x1::Float64, x2::Float64, y0::Float64, y1::Float64, y2::Float64)
	ret = y0 * (x - x1) / (x0 - x1) * (x - x2) / (x0 - x2)
	ret += y1 * (x - x0) / (x1 - x0) * (x - x2) / (x1 - x2)
	ret += y2 * (x - x0) / (x2 - x0) * (x - x1) / (x2 - x1)
	ret
end
	
function lagrange_polynomials(x::Float64, x0::Float64, x1::Float64, y0::Float64, y1::Float64)
	ret = y0 * (x - x1) / (x0 - x1) 
	ret += y1 * (x - x0) / (x1 - x0)
	ret
end

function lagrange_polynomials(x::Float64, x0::Float64, x1::Float64, y0::Vector{Float64}, y1::Vector{Float64})
	ret = y0 * (x - x1) / (x0 - x1) 
	ret += y1 * (x - x0) / (x1 - x0)
	ret
end

function fwdTriSolve!(A::SparseMatrixCSC, B::AbstractVecOrMat)
	# Todo
	# make fwdTriSolve!(X, A, B) which modifies X only.
	# forward substitution for CSC matrices
    n = length(B)
    if isa(B, Vector)
        nrowB = n
        ncolB = 1
    else
        nrowB, ncolB = size(B)
    end
    ncol = Base.LinAlg.chksquare(A)
    if nrowB != ncol
        throw(DimensionMismatch("A is $(ncol)X$(ncol) and B has length $(n)"))
    end

    aa = A.nzval
    ja = A.rowval
    ia = A.colptr

    joff = 0
    for k = 1:ncolB
        for j = 1:(nrowB-1)
            jb = joff + j
            i1 = ia[j]
            i2 = ia[j+1]-1
            B[jb] /= aa[i1]
            bj = B[jb]
            for i = i1+1:i2
                B[joff+ja[i]] -= bj*aa[i]
            end
        end
        joff += nrowB
        B[joff] /= aa[end]
    end
    return B
end



function hack_solve(A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64})
	#@assert istril(A)
	tmp = copy(b)
	fwdTriSolve!(A, tmp)
end



function method_trapezoidal(y0::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, h::Float64)
    # A is square.
    # 2nd order implicit.
    # y1 = y0 + 0.5 * h * (f(y0) + f(y1)
    # (I - 0.5hA) * y1 = (I + 0.5hA) * y0
	# local error = -1/12 * h^3 * y''' + O(h^4)
    I = speye(A)
    B = 0.5 * h * A

    rhs = (I + B) * y0

	hack_solve(I - B, rhs)
end



function method_Euler_implicit(
        y0::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, h::Float64
    )
    # A is square.
    # 1nd order implicit.
    # y1 = y0 + h * f(y1)
    # (I - hA) * y1 = y0
    # local error = -1/2 * h^2 * y'' + O(h^3)

    I = speye(A)
    B = h .* A

    hack_solve(I - B, y0)
end



function method_AB2!(y2::Vector{Float64}, tmp::Vector{Float64}, y0::Vector{Float64}, y1::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, h::Float64)
    # y2 is a place holder for output y2.
    # tmp is another place holder.
    # 2nd order 2-step Adams-Bashforth explicit.
    # y2 = y1 + 0.5 * h * (3 * f(y1) - f(y0))
	# local error = 5/12 * h^3 * y''' + O(h^4)

    # y2 = y1 + 0.5 * h * (3.0 * A * y1 - A * y0)
    
    for i in eachindex(y0)
        tmp[i] = 3.0 * y1[i] - y0[i]
    end
 
    y2 = A_mul_B!(y2, A, tmp)

    for i in eachindex(y0)
        y2[i] = y1[i] + y2[i] * 0.5 * h
    end   
    
    y2
end


function method_Euler!(y1::Vector{Float64}, y0::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, h::Float64)
    # y1 is a place holder for output y1.
    # 1st order 1-step Euler explicit.
    # y1 = y0 + h * f(y0)
    # local error = 1/2 * h^2 * y'' + O(h^3)
    
    y1 = A_mul_B!(y1, A, y0)
    for i in eachindex(y0)
        y1[i] = y0[i] + h * y1[i]
    end
    y1
end
