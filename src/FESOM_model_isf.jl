module FESOM_model_isf
function basic_simps(y::Array{Float64,1}, start::Int64, stop::Int64, dx::Float64)
    step = 2;
    slice0 = start:step:stop;
    slice1 = (start+1):step:(stop+1);
    slice2 = (start+2):step:(stop+2);
    # Even spaced Simpson's rule.
    
    result = sum(dx/3 * (y[slice0]+4*y[slice1]+y[slice2]));
    
    result
end
        
function basic_simps(y::Array{Float64,1}, start::Int64, stop::Int64, x::Array{Float64,1})
    step = 2;
    slice0 = start:step:stop;
    slice1 = (start+1):step:(stop+1);
    slice2 = (start+2):step:(stop+2);
    
    # Account for possibly different spacings.
    #    Simpson's rule changes a bit.
    h = x[2:end] - x[1:end-1];
    h0 = h[slice0];
    h1 = h[slice1];
    hsum = h0 .+ h1;
    hprod = h0 .* h1;
    h0divh1 = h0 ./ h1;
    tmp = (hsum ./ 6) .* (y[slice0] .* (2 .- 1.0 ./ h0divh1) .+
                      y[slice1] .* hsum .* hsum ./ hprod .+
                      y[slice2] .* (2 .- h0divh1));
    result = sum(tmp)
    result
end

function simps(y::Array{Float64,1};dx::Float64=1.0,even="avg")
    N = size(y,1);
    last_dx = dx;
    first_dx = dx;
    
    if N % 2 == 0
        if !(even in ["avg", "last", "first"])
            error("Parameter `even` must be `avg`, `last`, or `first`.");
        end

        val = 0.0;
        result = 0.0;
        
        # Compute using Simpson's rule on first intervals
        if even in ["avg", "first"]
            val += 0.5*last_dx*(y[end-1]+y[end]);
            result = basic_simps(y, 1, N-2, dx);
        end
        
        # Compute using Simpson's rule on last set of intervals
        if even in ["avg", "last"]
            val += 0.5*first_dx*(y[2]+y[1]);
            result += basic_simps(y, 2, N-1, dx);
        end
        
        if even == "avg"
            val /= 2;
            result /= 2;
        end
        result = result + val;
    else
        result = basic_simps(y, 1, N-1, dx);
    end
    
    result
end

function simps(y::Array{Float64,1}, x::Array{Float64,1}; even="avg")
    N = size(y,1);
    
    if size(x,1) != N
        error("`x` and `y` length must be equal");
    end
    
    if N % 2 == 0
        if !(even in ["avg", "last", "first"])
            error("Parameter `even` must be `avg`, `last`, or `first`.");
        end

        val = 0.0;
        result = 0.0;
        
        # Compute using Simpson's rule on first intervals
        if even in ["avg", "first"]
            last_dx = x[end] - x[end-1];
            val += 0.5*last_dx*(y[end-1]+y[end]);
            result = basic_simps(y, 1, N-2, x);
        end
        
        # Compute using Simpson's rule on last set of intervals
        if even in ["avg", "last"]
            first_dx = x[2] - x[1];
            val += 0.5*first_dx*(y[2]+y[1]);
            result += basic_simps(y, 2, N-1, x);
        end
        if even == "avg"
            val /= 2;
            result /= 2;
        end
        result = result + val;
    else
        result = basic_simps(y, 1, N-1, x);
    end
    result
end

function linear(isf_m::Array{Float64,1},dsf::Array{Float64,1},
                frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1})
    #FIXME
    println("zero T not implemented using workaround");
    beta = 10000000.0;
    linear(isf_m,dsf,frequency_bins,imaginary_time,beta);
    nothing
end

function linear(isf_m::Array{Float64,1},dsf::Array{Float64,1},
                frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1},beta::Float64)
    b = beta;
    
    @inbounds for i in 1:size(imaginary_time,1)
        t = imaginary_time[i];
        for j in 1:(size(frequency_bins,1) - 1)
            l = frequency_bins[j];
            m = frequency_bins[j + 1];
            n = (dsf[j + 1] - dsf[j])/(m - l);
            c = dsf[j] - n*l;
            if i == 1
                if j == 1
                    isf_m[i] = isf_linear(l,m,n,c,b);
                else
                    isf_m[i] += isf_linear(l,m,n,c,b);
                end
            else
                if j == 1
                    isf_m[i] = isf_linear(l,m,n,c,b,t);
                else
                    isf_m[i] += isf_linear(l,m,n,c,b,t);
                end
            end
        end
    end
    nothing
end

function isf_linear(l::Float64,m::Float64,n::Float64,c::Float64,b::Float64,t::Float64)
    ((exp(2*l*t)*(n + (c + l*n)*(b - t)))/(b - t)^2 + (exp(b*l)*(n + c*t + l*n*t))/t^2)/exp(l*(b + t)) + (-((exp(2*m*t)*(n + (c + m*n)*(b - t)))/(b - t)^2) - (exp(b*m)*(n + c*t + m*n*t))/t^2)/exp(m*(b + t))
end

function isf_linear(l::Float64,m::Float64,n::Float64,c::Float64,b::Float64)
    (l^2*n)/2 - (m^2*n)/2 - l*(c + l*n) + m*(c + m*n) + (n + b*(c + l*n))/(b^2*exp(b*l)) - (n + b*(c + m*n))/(b^2*exp(b*m))
end

function simps(isf_m::Array{Float64,1},dsf::Array{Float64,1},
                frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1})
    #FIXME
    println("zero T not implemented using workaround");
    beta = 10000000.0;
    simps(isf_m,dsf,frequency_bins,imaginary_time,beta);
    nothing
end

function simps(isf_m::Array{Float64,1},dsf::Array{Float64,1},
               frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1},beta::Float64)
    tmp = Array{Float64,1}(undef,size(dsf,1));
    @inbounds for i in 1:size(imaginary_time,1)
        t = imaginary_time[i];
        for j in 1:size(dsf,1)
            d = dsf[j];
            f = frequency_bins[j];
            tmp[j] = d*(exp(-t*f) + exp(-(beta - t)*f))/(1 + exp(-beta*f));
        end
        isf_m[i] = simps(tmp,frequency_bins)
    end
    nothing
end


end
