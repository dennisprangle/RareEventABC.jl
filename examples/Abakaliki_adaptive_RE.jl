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

##Markovian SIR binned to 5 day periods
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

##Do a pilot run, display some diagnostics, and return suggestions for main run: start point and random walk increment distribution
function pilot_run(model::RareEventABC.SEIRDetails, prior::Distribution, increment::Distribution, pars_init::Array{Float64, 1}, nparticles::Int; niterations=200)
    npars = length(pars_init)
    ##Run pilot
    pars, lls = doPMMH(model, eps, niterations, nparticles, prior, pars_init, increment)
    ##Plot output
    plot(lls, line=:steppost) |> display
    plot(pars, line=:steppost, layout=npars) |> display
    ##Diagnostics
    pilot_mean = vec(mean(pars, 1))
    print("Mean state $(round(pilot_mean, 2))\n")
    post_cov = cov(pars[100:niterations,:]) * (2.562^2 / 2)
    print("Recommended RW standard deviations $(round(sqrt(diag(post_cov)), 2))\n")
    print("Estimating log likelihood standard deviation...\n")
    llests = @showprogress Float64[doSMC(model, pilot_mean, eps, nparticles=nparticles, d=size_obs, quiet=true).llest for i in 1:50];
    print("Estimated log likelihood standard deviation $(round(std(llests), 2)) or $(round(std(llests[llests .> -Inf]), 2)) (removing $(sum(isinf(llests))) zero estimates)\n")
    return pilot_mean, MvNormal(zeros(npars), post_cov)
end

##Pilot runs. Not automated across models as tuning parameters are tweaked manually.
starts = Array(Array{Float64, 1}, 3);
increments = Array(Distribution, 3);

srand(1);
starts[1], increments[1] = pilot_run(sir_binned, prior_2,
                                     MvNormal(zeros(2), [0.05, 4.0]), ##RW increment
                                     [0.1, 10.0], ##Initial parameters
                                     400); ##Number of particles

srand(1);
starts[2], increments[2] = pilot_run(sir_gamma_I, prior_3,
                                     MvNormal(zeros(3), [0.02, 4.0, 2.0]), ##RW increment
                                     [0.1, 7.0, 2.5], ##Initial parameters
                                     300); ##Number of particles

srand(1);
starts[3], increments[3] = pilot_run(sir_weibull_P, prior_3,
                                     MvNormal(zeros(3), [0.05, 0.2, 2.0]), ##RW increment
                                     [0.1, 1.0, 14.0], ##Initial parameters
                                     200); ##Number of particles

main_nparticles = Int[400,300,200];

function main_run(model::RareEventABC.SEIRDetails, prior::Distribution, increment::Distribution, pars_init::Array{Float64, 1}, nparticles::Int; niterations=2000)
    npars = length(pars_init)
    ##PMMH run
    pars, lls = doPMMH(model, eps, niterations, nparticles, prior, pars_init, increment);
    ##Plot output
    plot(lls, line=:steppost) |> display
    plot(pars, line=:steppost, layout=npars) |> display
    ##Diagnostics
    print("Parameter means $(round(mean(pars, 1),2)) and standard deviations $(round(std(pars, 1),2))\n")
    return pars, lls
end

main_pars = Array(Array{Float64, 2}, 3);
main_lls = Array(Float64, (2000, 3));

srand(1);
for i in 1:3
    main_pars[i], main_lls[:,i] = main_run(models[i], priors[i], increments[i], vec(starts[i]), main_nparticles[i])
end

##PARAMETER ESTIMATES
for i in 1:3
    print("Means: $(round(mean(main_pars[i], 1), 3)) \n Sdev: $(round(std(main_pars[i], 1), 3)) \n")
end

##R0 ESTIMATES
R0_1 = main_pars[1][:,1] .* main_pars[1][:,2];
mean(R0_1), std(R0_1)

R0_2 = main_pars[2][:,1] .* main_pars[2][:,2] .* main_pars[2][:,3];
mean(R0_2), std(R0_2)

##ESTIMATE OF OTHER QUANTITIES
temp = main_pars[1][:,1]; ##Mean and standard deviation of pressure thresholds
mean(temp), std(temp)
temp = main_pars[1][:,2]; ##Mean and standard deviation of infectious period
mean(temp), std(temp)
temp = main_pars[2][:,1]; ##Mean and standard deviation of pressure thresholds
mean(temp), std(temp)
temp = main_pars[2][:,2] .* main_pars[2][:,3]; ##Mean of infectious period
mean(temp), std(temp)
temp = sqrt(main_pars[2][:,2]) .* main_pars[2][:,3]; ##Standard deviation of infectious period
mean(temp), std(temp)
temp = [mean(Weibull(main_pars[3][i,2],main_pars[3][i,1])) for i in 1:2000]; ##Mean of pressure thresholds
mean(temp), std(temp)
temp = [std(Weibull(main_pars[3][i,2],main_pars[3][i,1])) for i in 1:2000]; ##Standard deviation of pressure thresholds
mean(temp), std(temp)
temp = main_pars[3][:,3]; ##Mean and standard deviation of infectious period
mean(temp), std(temp)

