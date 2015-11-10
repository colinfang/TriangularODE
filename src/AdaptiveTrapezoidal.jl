module AdaptiveTrapezoidal
include("core_methods.jl")
using AdaptiveUtils
import AdaptiveEuler
export solveODE

function double_h_trapezoidal(y_next::Vector{Float64}, h::Float64, equation::Equation, history::History, info::ODE_Info)
	# h == t_next - t_current.
	# Have y(t+h), aim for y(t+h+2h), which requires y(t+h), y(t-h).
	# No need to interpolate, as long as y_1 is available and t_1 + h == t.
	t_current = top_t(history)
	t_1 = top_t(history, 1)

	ret = 
		if t_current == t_1 + h
			# double h
			t_next = t_current + h
			h_next = 2.0 * h
			
			# Modify t_current, y_current to be t_1, y_1.
			history = pop(history)
			history = push(history, t_next, y_next)
			
			equation = update(equation, h_next)
			log_double_h(info, t_next, h_next)
			h_next, equation, history
			
		else
			# keep h
			good_h_trapezoidal(y_next, h, equation, history, info)
		end	
	ret
end

function good_h_trapezoidal(y_next::Vector{Float64}, h::Float64, equation::Equation, history::History, info::ODE_Info)
	t_next = top_t(history) + h
	history = push(history, t_next, y_next)
	h, equation, history
end
		
function halve_h_trapezoidal(h::Float64, equation::Equation, history::History, info::ODE_Info)
	# to get y(t+0.5h), need y(t), y(t-0.5h)
	# Interpolate needed, from y_1, y_current.
	h_next = 0.5 * h
	equation = update(equation, h_next)
	t_current, y_current = top(history)
	t_1, y_1 = top(history, 1)
	
	adjusted_t_1 = t_current - h_next
	adjusted_y_1 = lagrange_polynomials(adjusted_t_1, t_1, t_current, y_1, y_current)
	
	log_halve_h(info, t_current, h_next)
	
	# Modify t_1, y_1
	history = pop(history)
	history = pop(history)
	history = push(history, adjusted_t_1, adjusted_y_1)
	history = push(history, t_current, y_current)	
	
	h_next, equation, history
end


function milne_device_trapezoidal!(
		y_tmp::Vector{Float64}, y_tmp2::Vector{Float64}, h::Float64, equation::Equation, history::History, tol::Tolerance
	)
	# y_tmp is the place holder to minimize memory allocation.
	# p_y_next can be mutable, as it is only needed inside the function.
	# Make sure y_next is immutable, as its value need to persist.
	
	y_current = top_y(history)
	y_1 = top_y(history, 1)
	# @assert h == equation.h
	p_y_next = method_AB2!(y_tmp, y_tmp2, y_1, y_current, equation.A, h)
	rhs = A_mul_B!(y_tmp2, equation.rhs_multiplier, y_current)
	y_next = hack_solve(equation.lhs, rhs)
	
	# Estimated local error of corrector.
	# May preallocate to save memory alloation.	
	# local_error = abs(p_y_next - y_next) / 6.0
	local_error = p_y_next
	for i in eachindex(y_next)
		local_error[i] = p_y_next[i] - y_next[i]
	end
	max_local_error = maxabs(local_error) / 6.0
	
	min_tolerance = get_min_tolerance(tol, h, y_next)
	
	ret = 
		if max_local_error * 10.0 < min_tolerance
			:double_h, y_next
		elseif max_local_error < min_tolerance
			:good_h, y_next
		else
			:halve_h, y_next
		end
	ret
end


function find_next_trapezoidal!(
		y_tmp::Vector{Float64}, y_tmp2::Vector{Float64}, h::Float64, equation::Equation, history::History, info::ODE_Info, tol::Tolerance
	)
	# For each h, there is an equation.
	# Find the next estimation, backtrack when failed.
	info.num_steps += 1
	flag, y_next = milne_device_trapezoidal!(y_tmp, y_tmp2, h, equation, history, tol)
	adjusted_history = history
	
	while flag == :halve_h
		# Backtrack to original history.
		h, equation, adjusted_history = halve_h_trapezoidal(h, equation, history, info)
		info.num_steps += 1
		flag, y_next = milne_device_trapezoidal!(y_tmp, y_tmp2, h, equation, adjusted_history, tol)
	end
	
	if flag == :double_h
		h, equation, history = double_h_trapezoidal(y_next, h, equation, adjusted_history, info)
	else
		h, equation, history = good_h_trapezoidal(y_next, h, equation, adjusted_history, info)
	end
	h, equation, history, info
end



function solveODE(y0::Vector{Float64}, t0::Float64, t_max::Float64, A::SparseMatrixCSC{Float64,Int64}; tol_abs::Float64=1e-6, tol_rel::Float64=1e-4)
	# tol are local error control per unit step.
	# Didn't check the error of the last step.

	# Initial guess.
	num_steps = 1000
	h = (t_max - t0) / num_steps
	
	history = initHistory(t0, y0)
	info = ODE_Info(h)
	
	# Get a place holder.
	y_tmp = copy(y0)
	y_tmp2 = copy(y0)
	tol = Tolerance(tol_abs, tol_rel)
	
	# First step has to be 1 step method.
	equation = EquationEulerImplicit(A, h)
	h, equation, history, info = AdaptiveEuler.find_next_Euler!(y_tmp, h, equation, history, info, tol)
	
	equation = EquationTrapezoidal(A, h)
	
	while check(h, history, t_max)
		h, equation, history, info = find_next_trapezoidal!(y_tmp, y_tmp2, h, equation, history, info, tol)		
	end
	
	info.num_steps += 1
	y_next = method_trapezoidal(top_y(history), A, t_max - top_t(history))
	@show info
	y_next
end

end