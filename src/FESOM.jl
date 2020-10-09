module FESOM
using UUIDs
using Random
using Statistics
using LinearAlgebra

include("./FESOM_model_isf.jl")

export fesom

function quality_of_fit(isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1})
    mean(broadcast((x,y,z) -> abs(x - y)/z,isf,isf_m,isf_error))
end

function quality_of_fit!(isf_m::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1})
    mean(broadcast!((x,y,z) -> abs(x - y)/z,isf_m,isf,isf_m,isf_error))
end

function fesom( dsf::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1};
                temperature::Float64 = 1.2,
                isf_m_type::String="simps",
                number_of_iterations::Int64=1000,
                stop_minimum_fitness::Float64 = 1.0e-8,
                seed::Int64=1)
    rng = MersenneTwister(seed);
    fesom(rng,dsf,isf,isf_error,frequency_bins,imaginary_time,
          temperature=temperature,
          isf_m_type=isf_m_type,
          number_of_iterations=number_of_iterations,
          stop_minimum_fitness=stop_minimum_fitness)
end

function fesom( rng::MersenneTwister,
                dsf::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1};
                temperature::Float64 = 1.2,
                isf_m_type::String="simps",
                number_of_iterations::Int64=1000,
                stop_minimum_fitness::Float64 = 1.0e-8)
    beta = 1/temperature;
    moment0 = isf[1];
    isf_m_function = getfield(FESOM_model_isf, Symbol(isf_m_type));
    random_vector = Array{Float64,1}(undef,size(dsf,1));
    dsf_new = Array{Float64,1}(undef,size(dsf,1));
    isf_m = Array{Float64,1}(undef,size(isf,1));

    isf_m_function(isf_m,dsf,frequency_bins,imaginary_time,beta);

    χ2 = quality_of_fit!(isf_m,isf,isf_error);
    χ2_new = 0.0;
    normalization = 1.0;
    minimum_fitness = zeros(number_of_iterations);
    for i in 1:number_of_iterations

        randn!(rng,random_vector);
        broadcast!((x,y) -> abs(x*(1 + y)),dsf_new, dsf, random_vector);

        #normalization: store intermediate results in random_vector
        normalization = FESOM_model_isf.simps(dsf_new,frequency_bins);
        broadcast!(*,dsf_new, dsf_new, moment0/normalization);

        isf_m_function(isf_m,dsf_new,frequency_bins,imaginary_time,beta);

        χ2_new = quality_of_fit!(isf_m,isf,isf_error);
        if χ2_new < χ2
            dsf_new, dsf = dsf, dsf_new;
            χ2_new, χ2 = χ2, χ2_new;
        end
        minimum_fitness[i] = χ2;
    end
    rng,dsf,χ2,minimum_fitness
end
end

