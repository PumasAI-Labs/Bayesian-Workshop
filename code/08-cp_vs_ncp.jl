using Pumas
using DataFrames
using CSV

# CP
poppk2cpt_cp = @model begin
    @param begin
        tvcl ~ LogNormal(log(10), 0.25)
        tvq ~ LogNormal(log(15), 0.5)
        tvvc ~ LogNormal(log(35), 0.25)
        tvvp ~ LogNormal(log(105), 0.5)
        tvka ~ LogNormal(log(2.5), 1)
        σ ~ truncated(Cauchy(0, 5), 0, Inf)
        C ~ LKJCholesky(5, 1.0)
        ω ∈ Constrained(
            MvNormal(zeros(5), Diagonal(0.4^2 * ones(5))),
            lower=zeros(5),
            upper=fill(Inf, 5),
            init=ones(5),
        )
    end

    @random begin
        η ~ MvNormal(ω .* C .* ω')
    end

    @pre begin
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

# NCP
poppk2cpt_ncp = @model begin
    @param begin
        tvcl ~ LogNormal(log(10), 0.25)
        tvq ~ LogNormal(log(15), 0.5)
        tvvc ~ LogNormal(log(35), 0.25)
        tvvp ~ LogNormal(log(105), 0.5)
        tvka ~ LogNormal(log(2.5), 1)
        σ ~ truncated(Cauchy(0, 5), 0, Inf)
        C ~ LKJCholesky(5, 1.0)
        ω ∈ Constrained(
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
pop = read_pumas(df)

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

# just 2 subjects
poppk2cpt_cp_fit = fit(
    poppk2cpt_cp,
    pop[1:2],
    iparams,
    Pumas.BayesMCMC(
        nsamples=200,
        nadapts=100,
        target_accept=0.6,
    )
)

poppk2cpt_ncp_fit = fit(
    poppk2cpt_ncp,
    pop[1:2],
    params,
    Pumas.BayesMCMC(
        nsamples=200,
        nadapts=100,
        target_accept=0.6,
    )
)

poppk2cpt_cp_tfit = Pumas.truncate(poppk2cpt_cp_fit; burnin=100)
poppk2cpt_ncp_tfit = Pumas.truncate(poppk2cpt_ncp_fit; burnin=100)

# comparing ESS
ess_cp = mean(ess(poppk2cpt_cp_tfit))
ess_ncp = mean(ess(poppk2cpt_ncp_tfit))
ess_ncp / ess_cp

# comparing Rhat
rhat_cp = mean(rhat(poppk2cpt_cp_tfit))
rhat_ncp = mean(rhat(poppk2cpt_ncp_tfit))
rhat_cp / rhat_ncp
