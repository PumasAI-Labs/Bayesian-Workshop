using Pumas
using DataFrames
using CSV

pk2cpt = @model begin
    @param begin
        tvcl ~ LogNormal(log(10), 0.25)
        tvq ~ LogNormal(log(15), 0.5)
        tvvc ~ LogNormal(log(35), 0.25)
        tvvp ~ LogNormal(log(105), 0.5)
        tvka ~ LogNormal(log(2.5), 1)
        σ ~ truncated(Cauchy(), 0, Inf)
    end

    @pre begin
        CL = tvcl
        Vc = tvvc
        Q = tvq
        Vp = tvvp
        Ka = tvka
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

df = CSV.read("data/pk2cpt.csv", DataFrame)
pop = read_pumas(df)

params = (;
    tvcl=7.4,
    tvq=28,
    tvvc=78,
    tvvp=68,
    tvka=1,
    σ=0.6
)

pk2cpt_fit = fit(
    pk2cpt,
    pop,
    params,
    Pumas.BayesMCMC(
        nsamples=2_000,
        nadapts=1_000,
    ),
)

Pumas.truncate(pk2cpt_fit; burnin=1_000)
