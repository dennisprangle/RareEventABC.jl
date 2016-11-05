using RareEventABC
using Plots; pgfplots()
default(guidefont=font(25), tickfont=font(25), legendfont=font(25), size=(500,500));
using StatPlots
using ProgressMeter
using DataFrames

##ABAKALIKI DATA
size_obs = 30;
popsize = 120;
robs = cumsum([0,13,7,2,3,0,0,1,4,5,3,2,0,2,0,5,3,1,4,0,1,1,1,2,0,1,5,0,5,5]);
duration_obs = maximum(robs);
robs = [robs; [Inf for i in 1:90]];

##MODELS
using RareEventABC
using Distributions

##Markoivan SIR binned to 5 day periods
sir_binned = RareEventABC.SEIRDetails(robs, popsize, θ->RareEventABC.Exponential(1.), θ->RareEventABC.DiracDelta(0.0), θ->Distributions.Exponential(θ[2]), x->floor(x/5.)*5.);

##Gamma infectious period
sir_gamma_I = RareEventABC.SEIRDetails(robs, popsize, θ->RareEventABC.Exponential(1.), θ->RareEventABC.DiracDelta(0.0), θ->Distributions.Gamma(θ[2], θ[3]), x->x);

##Weibull pressure thresholds
sir_weibull_P = RareEventABC.SEIRDetails(robs, popsize, θ->RareEventABC.Weibull(θ[2], 1.), θ->RareEventABC.DiracDelta(0.0), θ->Distributions.Exponential(θ[3]), x->x);

prior_2 = MvIndependent([Exponential(10.), Exponential(10.)]); ##Corresponds to Kypraios et al 2016 prior (n.b. Paper specified Exp(0.1) under rate parameterisation. Julia uses scale parameterisation.)
prior_3 = MvIndependent([Exponential(10.), Exponential(10.), Exponential(10.)]); ##Corresponds to Kypraios et al 2016 prior (n.b. Paper specified Exp(0.1) under rate parameterisation. Julia uses scale parameterisation.)

eps = 15.0;

priors = Distribution[prior_2, prior_3, prior_3];
models = RareEventABC.SEIRDetails[sir_binned, sir_gamma_I, sir_weibull_P];

"
Perform an ABC rejection analysis

Arguments:

* `model` is model to use
* `prior` is the prior distribution
* `N` is number of acceptances required
* `threshold` is threshold to investigate
"
function doABC(model::RareEventABC.SEIRDetails, prior::Distribution, N::Int, threshold::Float64)
    output_sample = zeros(Float64, (N,2))
    nsims = 0
    tic()
    @showprogress for i in 1:N
        while true
            prior_draw = rand(prior)
            next_dist = RareEventABC.getDistance(model, RareEventABC.simulateEpidemic(model, prior_draw))
            nsims += 1
            if next_dist <= threshold
                output_sample[i,:] = prior_draw
                break ##i.e. increment i in for loop
            end ##if
        end ##while
    end ##for
    timeABC = toq()
    time_per_sample = timeABC / N
    return time_per_sample
end ##trialABC

doABC(sir_binned, prior_2, 100, 100.0)




"
Run ABC-MCMC

Arguments:

* `model` is model to use
* `prior` is the prior distribution
* `nits` is number of iterations to perform
* `threshold` is threshold to investigate
* `increment` is random walk increment distribution
* `pars_init` is initial parameter values
"
function doABCMCMC(model::RareEventABC.SEIRDetails, prior::Distribution, nits::Int, threshold::Float64,
                   increment::Distribution, pars_init::Array{Float64, 1})
    tic()
    nacc = 0
    npars = length(pars_init)
    theta = pars_init
    sum_theta = theta
    sum_theta2 = theta .^2
    @showprogress for i in 2:nits
        theta_prop = theta + rand(increment)
        logratio = logpdf(prior, theta_prop) - logpdf(prior, theta)
        if (log(rand()) < logratio) ##1st step of acceptance, based on prior
            next_dist = RareEventABC.getDistance(model, RareEventABC.simulateEpidemic(model, theta_prop))
            if next_dist <= threshold
                theta = theta_prop ##Acceptance
                nacc += 1
            end
        end
        sum_theta .+= theta
        sum_theta2 .+= theta .^2
    end
    t = toq()
    mean_theta = sum_theta / nits
    std_theta = sqrt(max(0.0, sum_theta2./nits - mean_theta .^ 2))
    print("Mean thetas $(round(mean_theta, 2)), sdevs $(round(std_theta, 2)) \n")
    return DataFrame(threshold = threshold, nacc = nacc, accrate = nacc/nits)
end

inc1 = MvNormal([0.,0.], [0.00198571 -0.130246; -0.130246 23.1669]);
start1 = [0.11994, 10.4409];
doABCMCMC(sir_binned, prior_2, 10^7, 15.0, inc1, start1)

inc2 = MvNormal(zeros(3), [0.00139816   -0.113794    0.0590885;
                          -0.113794     19.9754    -10.3812;                           
                           0.0590885   -10.3812      7.47516]);
start2 = [0.103064, 4.11688, 4.28655];
doABCMCMC(sir_gamma_I, prior_3, 10^7, 15.0, inc2, start2)

inc3 = MvNormal(zeros(3), [0.00107213  0.006198   -0.0642286;                           
                           0.006198    0.0832468   0.0162699;
                          -0.0642286   0.0162699  10.1916]);
start3 = [0.0555384, 0.773309, 14.1865];
doABCMCMC(sir_gamma_I, prior_3, 10^7, 15.0, inc3, start3)
