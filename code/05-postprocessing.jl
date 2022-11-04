using Pumas
using DataFrames
using CSV
using PumasPlots

poppk2cpt = @model begin
    @param begin
        tvcl ~ LogNormal(log(10), 0.25)
        tvq ~ LogNormal(log(15), 0.5)
        tvvc ~ LogNormal(log(35), 0.25)
        tvvp ~ LogNormal(log(105), 0.5)
        tvka ~ LogNormal(log(2.5), 1)
        σ ~ truncated(Cauchy(0, 5), 0, Inf)
        C ~ LKJCholesky(5, 1.0)
        ω ~ Constrained(
            MvNormal(zeros(5), Diagonal(0.4^2 * ones(5))),
            lower=zeros(5),
            upper=fill(Inf, 5),
            init=ones(5),
        )
    end

    @random begin
        ηstd ~ MvNormal(I(5))
    end

    @pre begin
        # compute the η from the ηstd
        # using lower Cholesky triangular matrix
        η = ω .* (getchol(C).L * ηstd)

        # PK parameters
        CL = tvcl * exp(η[1])
        Q = tvq * exp(η[2])
        Vc = tvvc * exp(η[3])
        Vp = tvvp * exp(η[4])
        Ka = tvka * exp(η[5])
    end

    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL + Q) / Vc * Central + Q / Vp * Peripheral
        Peripheral' = Q / Vc * Central - Q / Vp * Peripheral
    end

    @derived begin
        cp := @. Central / Vc
        dv ~ @. LogNormal(log(cp), σ)
    end
end

df = CSV.read("data/poppk2cpt.csv", DataFrame)
pop = read_pumas(df)[1:5]

# Prior predictive check
prior_sims = simobs(poppk2cpt, pop[2]; samples = 200, simulate_error = true)
postprocess(prior_sims, stat = mean) do sim, data
    mean(sim.dv .> data.dv)
end

iparams = (;
    tvcl=9.5,
    tvq=19,
    tvvc=67,
    tvvp=102,
    tvka=1.2,
    σ=0.83,
    C=float.(Matrix(I(5))),
    ω=[0.8, 0.1, 1.8, 2.0, 0.5]
)

poppk2cpt_fit = fit(
    poppk2cpt,
    pop,
    iparams,
    Pumas.BayesMCMC(
        nsamples=1000,
        nadapts=500,
        target_accept = 0.6,
    )
)

poppk2cpt_tfit = Pumas.truncate(poppk2cpt_fit; burnin=500)

# Posterior Queries
mean(poppk2cpt_tfit) do p
    p.tvcl >= 1
end
mean(poppk2cpt_tfit; subject = 1) do p
    p.ηstd[2] <= 0
end
mean(poppk2cpt_tfit) do p
    getchol(p.C).L
end

# VPC
post_sims = simobs(poppk2cpt_tfit; samples = 100, simulate_error = true)
fit_vpc = vpc(post_sims)
vpc_plot(fit_vpc)
sim_plot(post_sims)

# Posterior predictive check
post_sims = simobs(poppk2cpt_tfit; samples = 100, simulate_error = false)
ps = postprocess(post_sims) do sim, data
    mean(sim.dv .> data.dv)
end
mean(ps)

# Simulate for new subject with new dose
subj_df = copy(df[df.id .== 1, :])
subj_df[1, :amt] = subj_df[1, :amt] * 3.0
new_subj = read_pumas(subj_df)[1]
new_sims = simobs(poppk2cpt_tfit, new_subj, samples = 100)
sim_plot(new_sims)
max_concs = postprocess(new_sims) do sim, data
    maximum(sim.dv)
end
mean(max_concs)
