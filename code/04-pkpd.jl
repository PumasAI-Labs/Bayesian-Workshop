using Pumas
using CSV
using DataFramesMeta

iv_2cmt_ir = @model begin
    @param begin
        tvcl ~ truncated(Cauchy(0, 2), 0, Inf)
        tvq ~ truncated(Cauchy(0, 2), 0, Inf)
        tvvc ~ truncated(Cauchy(0, 10), 0, Inf)
        tvvp ~ truncated(Cauchy(0, 20), 0, Inf)
        tvkin ~ truncated(Cauchy(0, 100), 0, Inf)
        tvkout ~ truncated(Cauchy(0, 2), 0, Inf)
        tvic50 ~ truncated(Cauchy(0, 5), 0, Inf)
        σ_p ~ truncated(Normal(0, 0.5), 0, Inf)
        σ_a ~ truncated(Normal(0, 1.0), 0, Inf)
        σ_PD ~ truncated(Normal(0, 0.4), 0, Inf)
        C ~ LKJCholesky(4, 2.0)
        ω ~ Constrained(
            MvNormal(zeros(4), Diagonal(0.4^2 * ones(4))),
            lower=zeros(4),
            upper=fill(Inf, 4),
            init=ones(4),
        )
        C_PD ~ LKJCholesky(3, 0.2)
        ω_PD ~ Constrained(
            MvNormal(zeros(3), Diagonal(0.4^2 * ones(3))),
            lower=zeros(3),
            upper=fill(Inf, 3),
            init=ones(3),
        )
    end

    @random begin
        ηstd ~ MvNormal(I(4))
        ηstd_PD ~ MvNormal(I(3))
    end

    @pre begin
        η = ω .* (getchol(C).L * ηstd)
        η_PD = ω_PD .* (getchol(C_PD).L * ηstd_PD)
        CL = tvcl * exp(η[1])
        Q = tvq * exp(η[2])
        Vc = tvvc * exp(η[3])
        Vp = tvvp * exp(η[4])
        Kin = tvkin * exp(η_PD[1])
        Kout = tvkout * exp(η_PD[2])
        ic50 = tvic50 * exp(η_PD[3])
    end

    @init begin
        Resp = Kin / Kout
    end

    @vars begin
        inh = (Central / Vc) / (ic50 + (Central / Vc))
    end

    @dynamics begin
        Central' = -(CL / Vc) * Central + (Q / Vp) * Peripheral - (Q / Vc) * Central
        Peripheral' = (Q / Vc) * Central - (Q / Vp) * Peripheral
        Resp' = (Kin * (1 - inh)) - (Kout * Resp) # idr: 0-order input and 1-order output
    end

    @derived begin
        conc := @. Central / Vc
        resp := Resp
        # censoring for lloq and bloq
        dv_PK ~ @. truncated(Normal(conc, sqrt(σ_a^2 + (conc * σ_p)^2)), 0, Inf)
        dv_PD ~ @. truncated(Normal(resp, σ_PD), 0, Inf)
    end
end

df = CSV.read("data/pkpd.csv", DataFrame)
pop = read_pumas(
    df;
    observations=[:dv_PK, :dv_PD]
)

iparams = (;
    tvcl=rand(LogNormal(log(0.3), 0.3)),
    tvvc=rand(LogNormal(log(5), 0.3)),
    tvq=rand(LogNormal(log(1), 0.3)),
    tvvp=rand(LogNormal(log(10), 0.3)),
    ω=rand(LogNormal(log(0.3), 0.3), 4),
    σ_a=rand(LogNormal(log(0.5), 0.3)),
    σ_p=rand(LogNormal(log(0.5), 0.3)),
    tvkin=rand(LogNormal(log(3), 0.3)),
    tvkout=rand(LogNormal(log(1), 0.3)),
    tvic50=rand(LogNormal(log(6), 0.3)),
    ω_PD=rand(LogNormal(log(0.3), 0.3), 3),
    σ_PD=rand(LogNormal(log(0.1), 0.3)),
    C=float.(Matrix(I(4))),
    C_PD=float.(Matrix(I(3))),
)

# fit
iv_2cmt_ir_fit = fit(
    iv_2cmt_ir,
    pop,
    iparams,
    Pumas.BayesMCMC(
        nsamples=100,
        nadapts=50,
        target_accept=0.5,
    )
)

tfit = Pumas.truncate(iv_2cmt_ir_fit; burnin=50)
println(DataFrame(summarystats(tfit)))
