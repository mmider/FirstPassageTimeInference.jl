include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                                Fetch the data
#------------------------------------------------------------------------------#

function read_τ_data(filename)
    datasets = nothing
    num_datasets = nothing
    open(filename, "r") do f
        for (i, line) in enumerate(eachline(f))
            if i == 1
                num_datasets = div(length(split(line, ",")), 2)
                datasets = [Vector{Tuple{Float64, Float64}}() for _ in 1:num_datasets]
            else
                data_line = map(x->parse(Float64, x), split(line, ","))
                for j in 1:num_datasets
                    append!(datasets[j], [(data_line[2*j-1], data_line[2*j])])
                end
            end
        end
    end
    datasets
end

data = read_τ_data(joinpath(OUT_DIR, "first_passage_times_hodgkin_huxley.csv"))

#------------------------------------------------------------------------------#
#                          Run inference on OU process
#------------------------------------------------------------------------------#
