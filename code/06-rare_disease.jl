using Pumas
using PumasPlots
using PharmaDatasets

bayes_model = @model begin
  @param begin
    # informative prior around the parameters
    disprog_placebo ~ Normal(0.0, 1.0)
    disprog_trt_effect ~ Normal(-0.2, 0.3)
    symptomatic_placebo ~ Normal(0.0, 1.5)
    symptomatic_trt_effect ~ Normal(0.2, 0.1)
    # mean(d::Gamma) = d.α * d.θ
    # var(d::Gamma) = mean(d) * d.θ
    ω²_1 ~ Gamma(1, 0.1) # mean = 0.1, var = 0.01
    ω²_2 ~ Gamma(1, 0.3) # mean = 0.3, var = 0.09
    kdelay ~ Gamma(20, 0.1) # mean = 2.0, var = 0.2
    σ² ~ Gamma(4, 0.25) # mean = 1.0, var = 0.25
  end
  @random begin
    bsv_dispro ~ Normal(0.0, sqrt(ω²_1))
    bsv_symp ~ Normal(0.0, sqrt(ω²_2))
  end
  @covariates trt
  @pre begin
    slope = disprog_placebo + disprog_trt_effect * trt + bsv_dispro
    symeff = symptomatic_placebo + symptomatic_trt_effect * trt + bsv_symp
    μ = slope * t - symeff * (1 - exp(-kdelay * t))
  end
  @derived begin
    ClinicalEndpoint ~ @. Normal(μ, sqrt(σ²))
  end
end

# Assume treatment works and has both symptomatic and disease progression effects
true_params = (
  disprog_placebo = 1.0,
  disprog_trt_effect = -0.5, # disprog_trt_effect < 0 means trt affects long-term disease progression
  symptomatic_placebo = 2.0,
  symptomatic_trt_effect = 0.3, # symptomatic_trt_effect > 0 means trt has short-term symptomatic effects
  ω²_1 = 0.1,
  ω²_2 = 0.3,
  kdelay = 2.0,
  σ² = 1.0,
)

trts = vcat(fill(0, 10), fill(1, 10))
sim_pop = map(1:20) do i
    Subject(; id = i, covariates = (trt = trts[i],), time = collect(1:1:26))
end
pop = Subject.(simobs(bayes_model, sim_pop, true_params))

iparams = init_params(bayes_model)
alg = MarginalMCMC(;
  marginal_alg = FOCE(),
  nsamples = 500,
  nadapts = 250,
  nchains = 4,
)
res = fit(bayes_model, pop, iparams, alg)
tres = Pumas.truncate(res; burnin = 250)

h = hpd(tres, alpha = 0.05)
disprog_trt_effect_credibint = (h[2, :lower], h[2, :upper])
symptomatic_trt_effect_credibint = (h[4, :lower], h[4, :upper])

# Post processing
# MCMC Chains
# getting parameter estimates
get_params(tres)

# Posterior Queries
mean(tres) do p
    p.disprog_trt_effect >= 0
end
mean(tres) do p
    p.symptomatic_trt_effect <= 0
end

# Visualizations
trace_plot(tres; parameters = [:disprog_trt_effect, :symptomatic_trt_effect])
density_plot(tres; parameters = [:disprog_trt_effect])
