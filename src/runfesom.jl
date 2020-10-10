using ArgParse
using Random
using JLD
using NPZ
using UUIDs

include("./FESOM.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--temperature", "-T"
            help = "Temperature of system."
            arg_type = Float64
            default = 0.0
        "--number_of_iterations", "-N"
            help = "Number of generations before genetic algorithm quits."
            arg_type = Int64
            default = 10000
        "--stop_minimum_fitness"
            help = "Regularization constant."
            arg_type = Float64
            default = 1.0e-8
        "--save_file_dir"
            help = "Directory to save results in."
            arg_type = String
            default = "./fesomresults"
        "--seed"
            help = "Seed to pass to random number generator."
            arg_type = Int64
            default = 1
        "--number_of_blas_threads"
            help = "Number of BLAS threads to utilize if > 0 otherwise uses Julia default value."
            arg_type = Int64
            default = 0
        "--track_stats"
            help = "Track minimum fitness."
            action = :store_true
        "qmc_data"
            help = "*.npz or *.jld file containing quantum Monte Carlo data with columns: IMAGINARY_TIME, INTERMEDIATE_SCATTERING_FUNCTION, ERROR"
            arg_type = String
            required = true
        "initial_dsf"
            help = "*.npz or *.jld file containing initial guess for dynamic structure factor and frequency with columns: DSF, FREQUENCY"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function main()
    start = time();
    parsed_args = parse_commandline()

    # create data directory
    try
        mkpath(parsed_args["save_file_dir"]);
    catch
        nothing
    end
    save_dir = parsed_args["save_file_dir"];

    _ext = splitext(parsed_args["qmc_data"])[2];
    if _ext == ".npz"
        qmcdata=NPZ.npzread(parsed_args["qmc_data"]);
    elseif _ext == ".jld"
        qmcdata = load(parsed_args["qmc_data"]);
    else
        throw(AssertionError("qmc_data must be *.jld or *.npz"));
    end
    imaginary_time = qmcdata["tau"];
    isf = qmcdata["isf"];
    isf_error = qmcdata["error"];

    _ext = splitext(parsed_args["initial_dsf"])[2];
    if _ext == ".npz"
        dsfdata=NPZ.npzread(parsed_args["initial_dsf"]);
    elseif _ext == ".jld"
        dsfdata = load(parsed_args["initial_dsf"]);
    else
        throw(AssertionError("initial_dsf must be *.jld or *.npz"));
    end
    dsf = dsfdata["dsf"];
    frequency_bins = dsfdata["frequency"];

    initial_dsf = copy(dsf);

    u4 = uuid4();
    rng = MersenneTwister(parsed_args["seed"]);

    rng,dsf_result,fitness, minimum_fitness = FESOM.fesom(rng,dsf,isf,isf_error,
                                                          frequency_bins,imaginary_time,
                                                          temperature = parsed_args["temperature"],
                                                          number_of_iterations = parsed_args["number_of_iterations"],
                                                          stop_minimum_fitness = parsed_args["stop_minimum_fitness"],
                                                          track_stats = parsed_args["track_stats"],
                                                          number_of_blas_threads = parsed_args["number_of_blas_threads"])
    elapsed = time() - start;
    seed = lpad(parsed_args["seed"],4,'0');
    filename = "$(save_dir)/fesom_results_$(seed)_$u4.jld";
    println("Saving results to $filename");
    save(filename,
         "u4",u4,
         "dsf",dsf_result,
         "frequency",frequency_bins,
         "fitness",fitness,
         "elapsed_time",elapsed);
    filename = "$(save_dir)/fesom_stats_$(seed)_$u4.jld";

    if parsed_args["track_stats"]
        println("Saving stats to $filename");
        save(filename,
             "u4",u4,
             "minimum_fitness",minimum_fitness);
    end

    filename = "$(save_dir)/fesom_params_$(seed)_$u4.jld";
    println("Saving parameters to $filename");
    save(filename,
         "u4",u4,
         "rng",rng,
         "imaginary_time",imaginary_time,
         "isf",isf,
         "isf_error",isf_error,
         "dsf",dsf_result,
         "frequency",frequency_bins,
         "initial_dsf",initial_dsf,
         "number_of_iterations",parsed_args["number_of_iterations"],
         "temperature",parsed_args["temperature"],
         "stop_minimum_fitness",parsed_args["stop_minimum_fitness"],
         "seed",parsed_args["seed"])
    nothing
end

main()
