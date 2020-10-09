module FESOM_covariance_methods
using LinearAlgebra
function eye(dsf::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1},
                temperature::Float64; covariance_options=Dict{String,Any}("type"=>"eye"))
    Matrix{Float64}(I, size(frequency_bins,1), size(frequency_bins,1))
end

function OrnsteinUhlenbeck(dsf::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1},
                           frequency_bins::Array{Float64,1},imaginary_time::Array{Float64,1},
                           temperature::Float64;
                           covariance_options=Dict{String,Any}("type"=>"OrnsteinUhlenbeck","alpha"=>0.1))
    #FIXME need to set alpha via https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation
    α::Float64=covariance_options["alpha"];
    cov = Matrix{Float64}(I, size(frequency_bins,1), size(frequency_bins,1));
    @inbounds for i = 1:size(frequency_bins,1)
        for j = 1:size(frequency_bins,1)
            cov[i,j] = exp(-α*abs(j - i))
        end
    end
    cov
end
end
