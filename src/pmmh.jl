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

function doPMMH(a::ApplicationDetails, schedule::Array{Float64, 1}, N::Int, nparticles::Int, prior::UnivariateDistribution, θinit::Float64, inc::UnivariateDistribution)
    states = zeros(Float64, N)
    loglikelihoods = zeros(Float64, N)
    states[1] = θ = θinit
    loglikelihoods[1] = ll = doSMC(a, [θ], schedule, nparticles=nparticles, quiet=true).llest
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            p = doSMC(a, [θprop], schedule, nparticles=nparticles, llBound=llBound, quiet=true)
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
            end
        end
        states[i], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods
end

function doPMMH(a::ApplicationDetails, schedule::Array{Float64, 1}, N::Int, nparticles::Int, prior::MultivariateDistribution, θinit::Array{Float64, 1}, inc::MultivariateDistribution)
    p = length(θinit)
    states = zeros(Float64, (N,p))
    loglikelihoods = zeros(Float64, N)
    states[1,:] = θ = copy(θinit)
    loglikelihoods[1] = ll = doSMC(a, θ, schedule, nparticles=nparticles, quiet=true).llest
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            p = doSMC(a, θprop, schedule, nparticles=nparticles, llBound=llBound, quiet=true)
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
            end
        end
        states[i,:], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods
end

function doPMMH(a::ApplicationDetails, targetε::Float64, N::Int, nparticles::Int, prior::UnivariateDistribution, θinit::Float64, inc::UnivariateDistribution; k=ceil(Int, nparticles/2))
    states = zeros(Float64, N)
    loglikelihoods = zeros(Float64, N)
    states[1] = θ = θinit
    loglikelihoods[1] = ll = doSMC(a, [θ], targetε, nparticles=nparticles, quiet=true, k=k).llest
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            p = doSMC(a, [θprop], targetε, nparticles=nparticles, llBound=llBound, quiet=true, k=k)
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
            end
        end
        states[i], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods
end
    

function doPMMH(a::ApplicationDetails, targetε::Float64, N::Int, nparticles::Int, prior::MultivariateDistribution, θinit::Array{Float64, 1}, inc::MultivariateDistribution; k=ceil(Int, nparticles/2))
    p = length(θinit)
    states = zeros(Float64, (N,p))
    loglikelihoods = zeros(Float64, N)
    states[1,:] = θ = copy(θinit)
    loglikelihoods[1] = ll = doSMC(a, θ, targetε, nparticles=nparticles, quiet=true, k=k).llest
    logprior = logpdf(prior, θ)
    @showprogress for i in 2:N
        θprop = θ + rand(inc)
        logpriorProp = logpdf(prior, θprop)
        llBound = log(rand()) + ll + logprior - logpriorProp
        if llBound < Inf
            p = doSMC(a, θprop, targetε, nparticles=nparticles, llBound=llBound, quiet=true, k=k)
            if !p.reachedBound ##Acceptance
                θ, ll = θprop, p.llest
            end
        end
        states[i,:], loglikelihoods[i] = θ, ll
    end
    states, loglikelihoods
end
