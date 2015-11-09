function fixed_step_trapezoidal(y0::Vector{Float64}, t0::Float64, t_max::Float64, A::SparseMatrixCSC{Float64,Int64}, num_steps::Int=50, error_calc::Bool=true)
	# Didn't check the error of the last step.
	# Didn't check the error of the first step.
	@assert num_steps > 10
	h = (t_max - t0) / num_steps
	t_current = t0
	t_next = t_current + h
	y_current = y0
	# Do one step.
	y_next = method_trapezoidal(y_current, A, h)
	
	# y_1 is the previous y.
	y_1, y_current = y_current, y_next
	t_current, t_next = t_next, t_next + h

	# Local errors per unit step.
	local_error_tot = 0.0
	local_error_rel = 0.0
	
	# Don't care about the local_error_rel if the value is too small.	
	relative_threshold = 1e-7
	#analytic_multiplier = expm(h * full(A))

	# This is the lhs of method_trapezoidal.	
	lhs = I - 0.5 * h * A

	# This is the lhs multiplier of method_trapezoidal.			
	rhs_multiplier = I + 0.5 * h * A
	
	
	while t_next < t_max
		y_next = hack_solve(lhs, rhs_multiplier * y_current)

		if error_calc
			p_y_next = method_AB2(y_1, y_current, A, h)
			# Estimated local error of corrector.
			# May preallocate to save memory alloation.
			local_error = abs(p_y_next - y_next) / 6.0
	
			#corrector_error = abs(analytic_multiplier * y_current - y_next)
			# That's predictor_error to predict correct_error, should be similar.
			#predictor_error = abs(analytic_multiplier * y_current - p_y_next)
	
			#@printf "%f, %f, %f\t" local_error[1] corrector_error[1] predictor_error[1]
			#@printf "%f, %f, %f\n" maximum(local_error) maximum(corrector_error) maximum(predictor_error) 
		
			local_error_tot = max(local_error_tot, maximum(local_error) / h)
			
			if local_error_rel >= 0.0
				rel_slice = abs(y_next) .> relative_threshold
				if isempty(rel_slice)
					local_error_rel = -1.0
				else
					local_error_rel = max(local_error_rel, maxabs((local_error ./ y_next)[rel_slice]) / h)
				end
			end
		end
		
		t_current, t_next = t_next, t_next + h
		y_1, y_current = y_current, y_next	
	end

	y_next = method_trapezoidal(y_current, A, t_max - t_current)
	y_next, local_error_tot, local_error_rel
end


function fixed_step_Euler(y0::Vector{Float64}, t0::Float64, t_max::Float64, A::SparseMatrixCSC{Float64,Int64}, num_steps::Int=50, error_calc::Bool=true)
	# This is used for benchmark and debug.
	# Didn't check the error of the last step.
	@assert num_steps > 10
	h = (t_max - t0) / num_steps
	t_current = t0
	t_next = t_current + h
	y_current = y0
		
	# Local errors per unit step.
	local_error_tot = 0.0
	local_error_rel = 0.0
	
	# Don't care about the local_error_rel if the value is too small.
	relative_threshold = 1e-7
	# Get a place holder.
	y_tmp = copy(y0)
	#analytic_multiplier = expm(h * full(A))

	# This is the lhs of method_Euler_implicit.
	lhs = I - h * A
	
	while t_next < t_max
		y_next = hack_solve(lhs, y_current)

		if error_calc
			# Be careful, p_y_next is essential y_tmp which is mutable.
			p_y_next = method_Euler!(y_tmp, y_current, A, h)
			# Estimated local error of corrector.
			# May preallocate to save memory alloation.
			local_error = abs(p_y_next - y_next) / 2.0
	
			#corrector_error = abs(analytic_multiplier * y_current - y_next)
			# That's predictor_error to predict correct_error, should be similar.
			#predictor_error = abs(analytic_multiplier * y_current - p_y_next)
	
			#@printf "%f, %f, %f\t" local_error[1] corrector_error[1] predictor_error[1]
			#@printf "%f, %f, %f\n" maximum(local_error) maximum(corrector_error) maximum(predictor_error) 
		
			local_error_tot = max(local_error_tot, maximum(local_error) / h)
			
			if local_error_rel >= 0.0
				rel_slice = abs(y_next) .> relative_threshold
				if isempty(rel_slice)
					local_error_rel = -1.0
				else
					local_error_rel = max(local_error_rel, maxabs((local_error ./ y_next)[rel_slice]) / h)
				end
			end
		end
		
		t_current, t_next = t_next, t_next + h
		y_current = y_next	
	end

	y_next = method_Euler_implicit(y_current, A, t_max - t_current)
	y_next, local_error_tot, local_error_rel
end