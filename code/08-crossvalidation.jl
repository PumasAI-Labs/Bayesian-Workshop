using Pumas
using PumasPlots
using PharmaDatasets

pkdata = dataset("iv_sd_3")
pop = read_pumas(pkdata)
# Just the first 10 subjects
pop = pop[1:10]

pk2cpt = @model begin
    @param begin
        # this is where we specify our priors
        tvcl ~ LogNormal(log(1.1), 0.25)
        tvvc ~ LogNormal(log(70), 0.25)
        σ ~ truncated(Cauchy(), 0, Inf)
        C ~ LKJCholesky(2, 1.0)
        ω ∈ Constrained(
            MvNormal(zeros(2), Diagonal(0.4^2 * ones(2))),
            lower = zeros(2),
            upper = fill(Inf, 2),
            init = ones(2),
        )
    end

    @random begin
        ηstd ~ MvNormal([1.0 0.0; 0.0 1.0])
    end

    @pre begin
        # compute the η from the ηstd
        # using lower Cholesky triangular matrix
        η = ω .* (getchol(C).L * ηstd)

        # PK parameters
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
    end

    @dynamics Central1

    @derived begin
        cp := @. Central / Vc
        dv ~ @. LogNormal(log(cp), σ)
    end
end

iparams = (; tvcl = 1, tvvc = 70, σ = 0.6, C = float.(Matrix(I(2))), ω = [0.1, 0.1])

# 4 chains as default
# Parallel as default across multiple subjects and chains
pk2cpt_fit = fit(pk2cpt, pop, iparams, Pumas.BayesMCMC(nsamples = 600, nadapts = 300))

# Remove the warmup/adapt/burn-in samples
tr_pk2cpt_fit = Pumas.truncate(pk2cpt_fit; burnin = 300)

cv_method = PSISCrossvalidation(
    split_method = LeaveK(K = 1),
    split_by = ByObservation(allsubjects = false),
);
r = crossvalidate(tr_pk2cpt_fit, cv_method);
elpd(r)

cv_method = PSISCrossvalidation(
    split_method = LeaveK(K = 1),
    split_by = BySubject(),
);
r = crossvalidate(tr_pk2cpt_fit, cv_method);
elpd(r)

cv_method = ExactCrossvalidation(
    split_method = KFold(K = 5),
    split_by = ByObservation(allsubjects = true),
);
r = crossvalidate(tr_pk2cpt_fit, cv_method);
elpd(r)

cv_method = ExactCrossvalidation(
    split_method = KFold(K = 5),
    split_by = BySubject(),
);
r = crossvalidate(tr_pk2cpt_fit, cv_method);
elpd(r)
