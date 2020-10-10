module FESOM
using UUIDs
using Random
using Statistics
using LinearAlgebra

export fesom

function quality_of_fit(isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1})
    mean(broadcast((x,y,z) -> ((x - y)/z)^2,isf,isf_m,isf_error))
end

function quality_of_fit!(isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1})
    mean(broadcast!((x,y,z) -> ((x - y)/z)^2,isf_m,isf,isf_m,isf_error))
end

function set_isf_term(imaginary_time::Array{Float64,1},frequency_bins::Array{Float64,1},beta::Float64)
    b = beta;
    isf_term = Array{Float64,2}(undef,size(imaginary_time,1),size(frequency_bins,1));
    for i in 1:size(imaginary_time,1)
        t = imaginary_time[i];
        for j in 1:size(frequency_bins,1)
            f = frequency_bins[j];
            isf_term[i,j] = (exp(-t*f) + exp(-(b - t)*f));
        end
    end
    isf_term
end

function set_isf_trapezoidal!( isf_m::Array{Float64,1}, isf_m2::Array{Float64,1},
                               isf_term::Array{Float64,2}, isf_term2::Array{Float64,2},
                               dsf::Array{Float64,1} 
                               )
    mul!(isf_m,isf_term,dsf);
    mul!(isf_m2,isf_term2,dsf);
    isf_m .+= isf_m2;
    nothing
end

function fesom( dsf::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                frequency::Array{Float64,1},imaginary_time::Array{Float64,1};
                temperature::Float64 = 1.2,
                number_of_iterations::Int64 = 10000,
                stop_minimum_fitness::Float64 = 1.0,
                seed::Int64 = 1,
                track_stats::Bool = false,
                number_of_blas_threads::Int64 = 0)
    rng = MersenneTwister(seed);
    fesom(rng,dsf,isf,isf_error,frequency,imaginary_time,
          temperature = temperature,
          number_of_iterations = number_of_iterations,
          stop_minimum_fitness = stop_minimum_fitness,
          number_of_blas_threads = number_of_blas_threads)
end

function fesom( rng::MersenneTwister,
                dsf::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                frequency::Array{Float64,1},imaginary_time::Array{Float64,1};
                temperature::Float64 = 1.2,
                number_of_iterations::Int64=10000,
                stop_minimum_fitness::Float64 = 1.0,
                track_stats::Bool = false,
                number_of_blas_threads::Int64 = 0)
    #_bts = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
    #println("Number of BLAS threads: $(_bts)");
    if number_of_blas_threads > 0
        LinearAlgebra.BLAS.set_num_threads(number_of_blas_threads)
    end

    beta = 1/temperature;
    moment0 = isf[1];
    random_vector = Array{Float64,1}(undef,size(dsf,1));
    dsf_new = Array{Float64,1}(undef,size(dsf,1));

    isf_term = set_isf_term(imaginary_time,frequency,beta);
    isf_term2 = copy(isf_term);

    df = frequency[2:end] .- frequency[1:size(frequency,1) - 1];
    dfrequency1 = zeros(size(frequency,1));
    dfrequency2 = zeros(size(frequency,1));
    for i in 1:(size(frequency,1) - 1)
        dfrequency1[i] = df[i]/2
        dfrequency2[i+1] = df[i]/2
    end
    isf_term .*= dfrequency1';
    isf_term2 .*= dfrequency2';
    
    isf_m = Array{Float64,1}(undef,size(isf,1));
    isf_m2 = Array{Float64,1}(undef,size(isf,1));

    set_isf_trapezoidal!(isf_m, isf_m2, isf_term, isf_term2, dsf );

    χ2 = quality_of_fit!(isf_m,isf,isf_error);
    χ2_new = 0.0;
    normalization = 1.0;

    if track_stats
        minimum_fitness = zeros(number_of_iterations);
    end
    nsteps = 0;
    for i in 1:number_of_iterations
        randn!(rng,random_vector);
        broadcast!((x,y) -> abs(x*(1 + y)),dsf_new, dsf, random_vector);

        #normalization: store intermediate results in random_vector
        normalization = dot(dsf_new,dfrequency1) + dot(dsf_new,dfrequency2);
        broadcast!(*,dsf_new, dsf_new, moment0/normalization);

        set_isf_trapezoidal!(isf_m, isf_m2, isf_term, isf_term2, dsf_new );

        χ2_new = quality_of_fit!(isf_m,isf,isf_error);
        if χ2_new < χ2
            dsf_new, dsf = dsf, dsf_new;
            χ2_new, χ2 = χ2, χ2_new;
        end

        nsteps +=1 ;

        if track_stats
            minimum_fitness[i] = χ2;
        end

        if χ2 < stop_minimum_fitness
            break
        end
    end

    if track_stats
        return rng,dsf,χ2,minimum_fitness[1:nsteps]
    else
        return rng,dsf,χ2,zeros(2)
    end
    
end
end

