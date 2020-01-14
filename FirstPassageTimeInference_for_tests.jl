_DIR = joinpath(Base.source_dir(), "src", "main")

function _load_in(filename; change_dir=nothing)
    global _DIR
    if change_dir !== nothing
        _DIR = joinpath(_DIR, "..", change_dir)
    end
    include(joinpath(_DIR, filename))
end

files = [
    "reposit.jl",
    "brownian_bridges.jl",
    "bessel.jl",
    "mcmc.jl",
    "random_walk.jl",
    "priors.jl",
]
for f in files _load_in(f) end


_load_in("ornstein_uhlenbeck.jl", change_dir="examples")
files = [
    "langevin_t.jl",
    "cox_ingersoll_ross.jl",
    "cox_ingersoll_ross_alt.jl",
    "conjug_updt_ou_cir.jl"
]
for f in files _load_in(f) end

_load_in("plot_summary.jl", change_dir="auxiliary")
_load_in("simulate_data.jl")

_load_in("cir_random_walk.jl",
         change_dir = joinpath("examples", "custom_kernels"))
_load_in("cir_alt_random_walk.jl")

OUT_DIR = joinpath(Base.source_dir(), "output")
mkpath(OUT_DIR)

SCRIPT_DIR = joinpath(Base.source_dir(), "scripts")
