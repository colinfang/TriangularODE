
function adaptive_step_Euler_draft(y0::Vector{Float64}, t0::Float64, t_max::Float64, A::SparseMatrixCSC{Float64,Int64}, tol_tot::Float64=1e-6, tol_rel::Float64=1e-4)
	# tol are local error control per unit step.
	# Didn't check the error of the last step.

	# Initial guess.
	num_steps = 1000
	h = (t_max - t0) / num_steps
	t_current = t0
	t_next = t_current + h
	y_current = y0
	
	# Info
	num_steps = 0
	min_h = h
	num_double_h = 0
	num_halve_h = 0	

	# Get a place holder.
	y_tmp = copy(y0)

	# This is the lhs of method_Euler_implicit.
	lhs = I - h * A
	
	while t_next < t_max
		num_steps += 1
		# Be careful, p_y_next is essential y_tmp which is mutable.
		p_y_next = method_Euler!(y_tmp, y_current, A, h)
		y_next = hack_solve(lhs, y_current)

		# Estimated local error of corrector.
		# May preallocate to save memory alloation.		
		local_error = abs(p_y_next - y_next) / 2.0
		# Got a feeling that order 1 method doesn't work well with error control per unit step.
		# As it converges too slow, halve h 3 times ~ 1 decimal place.
		tolerance = h * max(tol_tot, tol_rel * abs(y_next))
		
		max_local_error = maximum(local_error)
		min_tolerance = minimum(tolerance)
		
		# all(10.0 * local_error .< tolerance)
		if max_local_error * 10.0 < min_tolerance
			# Double step size.
			h *= 2.0
			t_current, t_next = t_next, t_next + h
			y_current = y_next
			lhs = I - h * A
			num_double_h += 1
			
			@show t_current
			println("Double h to")
			@show h
			@show num_steps
		elseif max_local_error < min_tolerance
			t_current, t_next = t_next, t_next + h
			y_current = y_next
			# Do nothing.
		else
			# Half step size.
			# Redo current step.
			t_next -= h
			h *= 0.5
			t_next += h
			lhs = I - h * A
			num_halve_h += 1
			min_h = min(h, min_h)
			
			@show t_current
			println("Halve h to")
			@show h
			@show num_steps
		end
	end
	num_steps += 1
	y_next = method_Euler_implicit(y_current, A, t_max - t_current)
	@show num_steps
	@show min_h
	@show num_double_h
	@show num_halve_h
	y_next
end