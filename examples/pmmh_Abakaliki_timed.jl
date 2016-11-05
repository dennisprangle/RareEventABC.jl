##COMPARE TIME TAKEN BY ADAPTIVE AND NON-ADAPTIVE RE-ABC IN ABAKALIKI EXAMPLE
using RareEventABC
using Distributions
using DataFrames
using ProgressMeter
using Plots; pgfplots();
using StatPlots;

"
A pseudo-marginal Metropolis-Hastings algorithm which uses RE-ABC likelihood estimates.

## Arguments

- `a` Details of the application.
- `schedule` Sequence of epsilon values. Use this option for non-adaptive RE-ABC.
- `targetε` Target ε value. Use this option for adaptive RE-ABC.
- `N` Number of iterations to do.
- `nparticles` Number of particles to use.
- `prior` Prior distribution.
- `θinit` Initial state of chain.
- `inc` Distribution of proposal increments
- `k` Number of particles to accept at each iteration of adaptive RE-ABC.
"
function timedPMMH(a::RareEventABC.ApplicationDetails, schedule::Array{Float64, 1}, N::Int, nparticles::Int, prior::MultivariateDistribution, θinit::Array{Float64, 1}, inc::MultivariateDistribution)
    p = length(θinit)
    states = zeros(Float64, (N,p))
    loglikelihoods = zeros(Float64, N)
    states[1,:] = θ = copy(θinit)
    times = DataFrame(time=Float64[], stop=String[])    
    tic()
    loglikelihoods[1] = ll = doSMC(a, θ, schedule, nparticles=nparticles, quiet=true).llest
    t = toq()
    push!(times, (t, "complete"))    
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            tic()
            p = doSMC(a, θprop, schedule, nparticles=nparticles, llBound=llBound, quiet=true)
            t = toq()
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
                push!(times, (t, "complete"))
            else
                push!(times, (t, "early"))   
            end
        end
        states[i,:], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods, times
end

function timedPMMH(a::RareEventABC.ApplicationDetails, targetε::Float64, N::Int, nparticles::Int, prior::MultivariateDistribution, θinit::Array{Float64, 1}, inc::MultivariateDistribution; k=ceil(Int, nparticles/2))
    p = length(θinit)
    states = zeros(Float64, (N,p))
    loglikelihoods = zeros(Float64, N)
    states[1,:] = θ = copy(θinit)
    times = DataFrame(time=Float64[], stop=String[])
    tic()
    loglikelihoods[1] = ll = doSMC(a, θ, targetε, nparticles=nparticles, quiet=true, k=k).llest
    t = toq()
    push!(times, (t, "complete"))
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            tic()
            p = doSMC(a, θprop, targetε, nparticles=nparticles, llBound=llBound, quiet=true, k=k)
            t = toq()
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
                push!(times, (t, "complete"))
            else
                push!(times, (t, "early"))
            end
        end
        states[i,:], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods, times
end

##ABAKALIKI DATA
size_obs = 30;
popsize = 120;
robs = cumsum([0,13,7,2,3,0,0,1,4,5,3,2,0,2,0,5,3,1,4,0,1,1,1,2,0,1,5,0,5,5]);
duration_obs = maximum(robs);
robs = [robs; [Inf for i in 1:90]];

##Model (SIR with gamma infectious period) and prior
sir_gamma_I = RareEventABC.SEIRDetails(robs, popsize, θ->RareEventABC.Exponential(1.), θ->RareEventABC.DiracDelta(0.0), θ->Distributions.Gamma(θ[2], θ[3]), x->x);
prior = MvIndependent([Exponential(10.), Exponential(10.), Exponential(10.)]);

##Tuning choices (taken from original analysis)
start = [0.103064, 4.11688, 4.28655];
inc = MvNormal(zeros(3), [0.00139816   -0.113794    0.0590885;
                         -0.113794     19.9754    -10.3812;                           
                          0.0590885   -10.3812      7.47516]);

##NON-ADAPTIVE RE-ABC
##Tuning
srand(1);
schedule = doSMC(sir_gamma_I, start, 15.0, nparticles=300, d=size_obs, quiet=false).schedule;

##Main run
thetas, ll, times = timedPMMH(sir_gamma_I, schedule, 2000, 300, prior, start, inc);

##ADAPTIVE RE-ABC
thetas2, ll2, times2 = timedPMMH(sir_gamma_I, 15.0, 2000, 300, prior, start, inc);

alltimes = vcat(hcat(times, fill("non-adaptive", size(times)[1])), hcat(times2, fill("adaptive", size(times2)[1])))
rename!(alltimes, :x1, :method)

writetable("pmmh_Abakaliki_timed.csv", alltimes) ##To plot using R

