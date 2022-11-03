using Pumas
using DataFrames
using StableRNGs
using AlgebraOfGraphics
using CairoMakie

# Generate funnel data
function simulate_data(; nsamples=100, rng=StableRNG(123))
    x = rand(rng, Normal(), nsamples)
    y = rand.(rng, Normal.(0, exp.(x ./ 2)))
    return DataFrame(; id=1:nsamples, x, y, time=0, amt=0)
end

df = simulate_data()

# Plot the funnel
data(df) *
mapping(:y, :x) *
visual(Scatter) |> draw

# Pumas
pop = read_pumas(df; observations=[:y], covariates=[:x], event_data=false)

cp = @model begin
    @param begin
        x ~ Normal()
    end
    @derived begin
        y ~ @. Normal(0, exp(x / 2))
    end
end

ncp = @model begin
    @param begin
        x ~ Normal()
        ystd ~ Normal()
    end
    @derived begin
        y = @. ystd * exp(x / 2)
    end
end

fit_cp = fit(
    cp,
    pop,
    init_params(cp),
    Pumas.BayesMCMC(
        nsamples=2_000,
        nadapts=1_000,
    ),
)

tfit_cp = Pumas.truncate(cp_fit; burnin=1_000)

fit_ncp = fit(
    ncp,
    pop,
    init_params(ncp),
    Pumas.BayesMCMC(
        nsamples=2_000,
        nadapts=1_000,
    ),
)

tfit_cp = Pumas.truncate(cp_fit; burnin=1_000)
