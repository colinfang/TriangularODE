module AdaptiveUtils
using DataStructures
export ODE_Info, log_double_h, log_halve_h, History, top_t, top_y, top, push, pop, initHistory, check, Equation
export EquationEulerImplicit, EquationTrapezoidal, update, get_tolerance, get_min_tolerance, Tolerance, EquationAM2

type ODE_Info
	num_steps::Int
	min_h::Float64
	num_double_h::Int
	num_halve_h::Int
	#t_at_double::Vector{Float64}
	#t_at_halve::Vector{Float64}
end

ODE_Info(min_h::Float64) = ODE_Info(0,min_h,0,0)

function log_double_h(info::ODE_Info, t::Float64, h::Float64)
	# t is the last one before change of h.
	# h is the new h.
	info.num_double_h += 1
	@printf "At step %d, t = %f, will double h to %f for the next step.\n" info.num_steps t h
end

function log_halve_h(info::ODE_Info, t::Float64, h::Float64)
	# t is the last one before change of h.
	# h is the new h.
	info.num_halve_h += 1
	info.min_h = min(info.min_h, h)
	@printf "At step %d, t = %f, retry to halve h to %f for the next step.\n" info.num_steps t h	
end


immutable History
	# It keeps history of items of equal distance.
	# It is supposed to be a immutable data structure.
	t::LinkedList{Float64}
	y::LinkedList{Vector{Float64}}
	count::Int
end

top_t(x::History) = head(x.t)
top_y(x::History) = head(x.y)
top(x::History) = head(x.t), head(x.y)
# top(x) = top(x, 0)

function top_t(x::History, n::Int)
	xs = x.t
	while n > 0
		xs = tail(xs)
		n -= 1
	end
	head(xs)
end

Base.length(x::History) = x.count


function top_y(x::History, n::Int)
	xs = x.y
	while n > 0
		xs = tail(xs)
		n -= 1
	end
	head(xs)
end

		
function top(x::History, n::Int)
	ts, ys = x.t, x.y
	while n > 0
		ts = tail(ts)
		ys = tail(ys)
		n -= 1
	end
	head(ts), head(ys)
end

function push(x::History, t, y)
	History(cons(t, x.t), cons(y, x.y), x.count + 1)
end

function pop(x::History)
	# Returns the remaining.
	# Do not return the popped value.
	# Exception if failed.
	History(tail(x.t), tail(x.y), x.count - 1)
end


function initHistory(t::Float64, y::Vector{Float64})
	ts = nil(Float64)
	ys = nil(Vector{Float64})
	ts = cons(t, ts)
	ys = cons(y, ys)
	History(ts, ys, 1)
end


# Loop condition, do not exceed max t.
check(h::Float64, history::History, t_max::Float64) = h + top_t(history) < t_max


abstract Equation

immutable EquationEulerImplicit <: Equation
	A::SparseMatrixCSC{Float64,Int64}
	lhs::SparseMatrixCSC{Float64,Int64}
	h::Float64
end

EquationEulerImplicit(A::SparseMatrixCSC{Float64,Int64}, h::Float64) = EquationEulerImplicit(A, I - h * A, h)
update(equation::EquationEulerImplicit, h::Float64) = EquationEulerImplicit(equation.A, h)


immutable EquationTrapezoidal <: Equation
	A::SparseMatrixCSC{Float64,Int64}
	lhs::SparseMatrixCSC{Float64,Int64}
	rhs_multiplier::SparseMatrixCSC{Float64,Int64}
	h::Float64
end

function EquationTrapezoidal(A::SparseMatrixCSC{Float64,Int64}, h::Float64)
	B = 0.5 * h * A
	lhs = I - B
	rhs_multiplier = I + B
	EquationTrapezoidal(A, lhs, rhs_multiplier, h)
end
update(equation::EquationTrapezoidal, h::Float64) = EquationTrapezoidal(equation.A, h)


immutable EquationAM2 <: Equation
	A::SparseMatrixCSC{Float64,Int64}
	lhs::SparseMatrixCSC{Float64,Int64}
	rhs_multiplier_current::SparseMatrixCSC{Float64,Int64}
	rhs_multiplier_1::SparseMatrixCSC{Float64,Int64}
	h::Float64
end

function EquationAM2(A::SparseMatrixCSC{Float64,Int64}, h::Float64)
	lhs = I - 5.0 / 12.0 * h * A
	# y_current
	rhs_multiplier_current = I + 2.0 / 3.0 * h * A
	# y_1
	rhs_multiplier_1 = -1.0 / 12.0 * h * A
	EquationAM2(A, lhs, rhs_multiplier_current, rhs_multiplier_1, h)
end
update(equation::EquationAM2, h::Float64) = EquationAM2(equation.A, h)



immutable Tolerance
	abs::Float64
	rel::Float64
end

get_tolerance(x::Tolerance, h::Float64, y::Vector{Float64}) = h * max(x.abs, x.rel * abs(y))


function get_min_tolerance(x::Tolerance, h::Float64, y::Vector{Float64})
	# tolerance = get_tolerance(tol, h, y_next)
	# min_tolerance = minimum(tolerance)
	# The following is an optimized version.
	a = minabs(y)
	ret = max(x.abs, x.rel * a)
	ret * h
end

end