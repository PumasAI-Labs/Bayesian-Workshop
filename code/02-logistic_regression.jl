using Pumas
using PharmaDatasets
using DataFramesMeta

nausea = dataset("nausea")

# 1. Define the model
logistic_model = @model begin
    @param begin
        α ~ Normal(0, 2.5)
        βAUC ~ Normal(0, 2.5)
        βisF ~ Normal(0, 2.5)
        βRACE_Caucasian ~ Normal(0, 2.5)
    end

    @covariates begin
        AUC
        isF
        RACE
    end

    @pre begin
        _AUC = AUC * βAUC
        _isF = isF * βisF
        _RACE_Causasian = (RACE == "Caucasian") * βRACE_Caucasian
        linear_pred = _AUC + _isF + _RACE_Causasian
    end

    @derived begin
        NAUSEA ~ @. Bernoulli(logistic(α + linear_pred))
    end
end

# 2. Read data into a population
# create a column of 0s as :time
@rtransform! nausea :time = 0
pop = read_pumas(
    nausea;
    observations=[:NAUSEA],
    covariates=[:AUC, :isF, :RACE],
    id=:ID,
    time=:time,
    event_data=false
)

# 3. Fit the model
nausea_fit = fit(
    logistic_model,
    pop,
    init_params(logistic_model),
    Pumas.BayesMCMC()
)
