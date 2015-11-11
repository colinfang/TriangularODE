module AdaptiveEuler
include("core_methods.jl")
using AdaptiveUtils
export solveODE

function double_h(y_next::Vector{Float64}, h::Float64, equation::EquationEulerImplicit, history::History, info::ODE_Info)	
	t_next = top_t(history) + h
	history = push(history, t_next, y_next)
	
	h_next = 2.0 * h
	equation = update(equation, h_next)
	
	log_double_h(info, t_next, h_next)
	h_next, equation, history
end

function good_h(y_next::Vector{Float64}, h::Float64, equation::EquationEulerImplicit, history::History, info::ODE_Info)
	t_next = top_t(history) + h
	history = push(history, t_next, y_next)
	h, equation, history
end
		
function halve_h(h::Float64, equation::EquationEulerImplicit, history::History, info::ODE_Info)	
	h_next = 0.5 * h
	equation = update(equation, h_next)
	t_current = top_t(history)
	
	log_halve_h(info, t_current, h_next)
	h_next, equation, history
end


function milne_device!(
		y_tmp::Vector{Float64}, h::Float64, equation::EquationEulerImplicit, history::History, tol::Tolerance
	)
	# y_tmp is the place holder to minimize memory allocation.
	# p_y_next can be mutable, as it is only needed inside the function.
	# Make sure y_next is immutable, as its value need to persist.
	
	y_current = top_y(history)
	# @assert h == equation.h
	p_y_next = method_Euler!(y_tmp, y_current, equation.A, h)
	y_next = hack_solve(equation.lhs, y_current)
	
	# Estimated local error of corrector.
	# May preallocate to save memory alloation.
	# local_error = abs(p_y_next - y_next) / 2.0
	local_error = p_y_next
	for i in eachindex(y_next)
		local_error[i] = p_y_next[i] - y_next[i]
	end
	max_local_error = maxabs(local_error) / 2.0
	
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


function find_next!(
		y_tmp::Vector{Float64}, h::Float64, equation::EquationEulerImplicit, history::History, info::ODE_Info, tol::Tolerance;
		allow_double_h::Bool=true
	)
	# For each h, there is an equation.
	# Find the next estimation, backtrack when failed.
	info.num_steps += 1
	flag, y_next = milne_device!(y_tmp, h, equation, history, tol)
	adjusted_history = history
	
	while flag == :halve_h
		# Backtrack to original history.
		h, equation, adjusted_history = halve_h(h, equation, history, info)
		info.num_steps += 1
		flag, y_next = milne_device!(y_tmp, h, equation, adjusted_history, tol)
	end
	
	if flag == :double_h && allow_double_h
		h, equation, history = double_h(y_next, h, equation, adjusted_history, info)
	else
		h, equation, history = good_h(y_next, h, equation, adjusted_history, info)
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
	equation = EquationEulerImplicit(A, h)
	
	# Get a place holder.
	y_tmp = copy(y0)
	tol = Tolerance(tol_abs, tol_rel)
	while check(h, history, t_max)
		h, equation, history, info = find_next!(y_tmp, h, equation, history, info, tol)		
	end
	
	info.num_steps += 1
	y_next = method_Euler_implicit(top_y(history), A, t_max - top_t(history))
	@show info
	y_next
end

end