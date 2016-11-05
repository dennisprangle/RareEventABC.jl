##COMPARE TIME TAKEN BY ADAPTIVE AND NON-ADAPTIVE RE-ABC IN NORMAL EXAMPLE
using RareEventABC
using Distributions
using DataFrames
using ProgressMeter
using Plots; pyplot();
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

function timedPMMH(a::RareEventABC.ApplicationDetails, schedule::Array{Float64, 1}, N::Int, nparticles::Int, prior::UnivariateDistribution, θinit::Float64, inc::UnivariateDistribution)
    states = zeros(Float64, N)
    loglikelihoods = zeros(Float64, N)
    states[1] = θ = θinit
    times = DataFrame(time=Float64[], early=Bool[])
    tic()
    loglikelihoods[1] = ll = doSMC(a, [θ], schedule, nparticles=nparticles, quiet=true).llest
    t = toq()
    push!(times, (t, false))
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            tic()
            p = doSMC(a, [θprop], schedule, nparticles=nparticles, llBound=llBound, quiet=true)
            t = toq()
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
                push!(times, (t, false))
            else
                push!(times, (t, true))
            end
        end
        states[i], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods, times
end

function timedPMMH(a::RareEventABC.ApplicationDetails, targetε::Float64, N::Int, nparticles::Int, prior::UnivariateDistribution, θinit::Float64, inc::UnivariateDistribution; k=ceil(Int, nparticles/2))
    states = zeros(Float64, N)
    loglikelihoods = zeros(Float64, N)
    states[1] = θ = θinit
    times = DataFrame(time=Float64[], early=Bool[])
    tic()
    loglikelihoods[1] = ll = doSMC(a, [θ], targetε, nparticles=nparticles, quiet=true, k=k).llest
    t = toq()
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            tic()
            p = doSMC(a, [θprop], targetε, nparticles=nparticles, llBound=llBound, quiet=true, k=k)
            t = toq()
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
                push!(times, (t, false))
            else
                push!(times, (t, true))
            end
        end
        states[i], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods, times
end

##Generate observations
srand(1);
d = 25;
yobs = randn(d) .* 3.0;
a = RareEventABC.IIDNormalDetails(yobs, d);

rmse(x::Vector, y::Float64) = (x .- y) .^ 2 |> mean |> sqrt;
rmse(x::Vector, y::Float64, w::Vector) = (w .* ((x .- y) .^ 2)) |> sum |> a->a/sum(w) |> sqrt;

##NON-ADAPTIVE RE-ABC
##Tuning
schedule = doSMC(a, [3.0], 5.0, nparticles=300, quiet=false).schedule;
thetas, ll, times = timedPMMH(a, schedule, 200, 300, Uniform(0.0,10.0), 3.0, Normal(0.0,0.5));
sd_hd = sqrt(2.562) * std(thetas)
##Main run
thetas, ll, times = timedPMMH(a, schedule, 2000, 300, Uniform(0.0,10.0), 3.0, Normal(0.0,sd_hd));

##ADAPTIVE RE-ABC
thetas2, ll2, times2 = timedPMMH(a, 5.0, 2000, 300, Uniform(0.0,10.0), 3.0, Normal(0.0,sd_hd));

alltimes = vcat(hcat(times, fill("non-adaptive", size(times)[1])), hcat(times2, fill("adaptive", size(times2[1]))));
rename!(alltimes, :x1, :method);          

writetable("pmmh_normal_timed.csv", alltimes) ##To plot (using R)

isearly = alltimes[:early].data;
isadaptive = alltimes[:method].data == "adaptive";

default(xlim=(0., 0.26), ylim=(0, 250), legend=false)
p1 = histogram(alltimes[isearly  & isadaptive, :time], xlim=(0.,0.26), title="Adaptive, stopped early");
p2 = histogram(alltimes[!isearly & isadaptive, :time], xlim=(0.,0.26), title="Adaptive, completed");
p3 = histogram(alltimes[isearly  & !isadaptive, :time], xlim=(0.,0.26), title="Non-adaptive, stopped early");
p4 = histogram(alltimes[!isearly & !isadaptive, :time], xlim=(0.,0.26), title="Non-adaptive, completed");

plot(p1,p2,p3,p4,layout=4)
