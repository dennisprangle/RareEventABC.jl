###################################
##General ABC SMC algorithm
###################################

"
Details of an application, mainly the data and model of interest.
The following related methods should be defined.

- `getDistances(a::ApplicationDetails, particles::Particles)`
- `report(a::ApplicationDetails, p::Particles, targetε::Float64)`
- `updateParticles!(a::ApplicationDetails, p::Particles, θ::Array{Float64, 1})`
"
abstract ApplicationDetails

"
Details of the current weighted particles during an SMC run.
This should include fields:

- `nparticles` Number of particles
- `llest` Current log likelihood estimate
- `ε` Current value of epsilon
- `iteration` How many SMC iterations have been performed
- `reachedBound` True if ABChd has terminated early because a lower bound specified on the log likelihood has been reached.
- `schedule` Schedule of epsilon values used (or, for an adaptive algorithm, schedule so far)
- `maxz` Maximum slice sampling value of |z| in previous SMC iteration

The following related methods should be defined:

- `Particles(a::AlgorithmDetails, nparticles::Int, parameters::Array{Float64, 1})`, a constructor. This should initialise each particle from the model specified by `a` under specified `parameters` value.
- `shuffle!(particles::Particles, indices::Array{Int, 1})` Updates particles so ith new particle is a copy of indices[i]th old particle.
"
abstract Particles

"
Performs or continues an SMC run.

## Arguments

- `a` Details of the application.
- `parameters` Vector of parameters whose likelihood is to be estimated.
- `schedule` Schedule of ε values. (Only required in non-adaptive version.)
- `targetε` Target ε value. (Only required in adaptive version.)
- `nparticles` Number of particles to use
- `k` Number of particles to accept at each iteration. (Only required in adaptive version.)
- `llBound` Lower bound on the log-likelihood. The algorithm terminates if the estimate is guaranteed to be lower than this.
- `d` (Effective) data dimension. Used in the normalising constant of the likelihood estimate, so optional.
- `quiet` When `false` progress reports are shown.

## Returns a `Particles` object with all the final information
"
##Non-adaptive version:
function doSMC(a::ApplicationDetails, parameters::Array{Float64, 1}, schedule::Array{Float64, 1}; nparticles=200, llBound=-Inf, d=0, quiet=false)
    ##Initialise SMC, including sampling from prior
    particles::Particles = Particles(a, nparticles, parameters)
    particles.schedule = schedule
    particles.llest = lgamma(1+d/2)-(d/2)*log(pi)-d*log(schedule[end]) ##log(1/V) where V is Lebesgue measure of a radius schedule[end] and dimension d
    particles.iteration = 0 ##Number of *completed* iterations

    ##Main loop
    nits = length(schedule)
    for ε in schedule
        particles.ε = ε
        dists::Array{Float64, 1} = getDistances(a, particles)
        acc = (dists .<= ε)
        particles.llest += log(mean(acc))
        particles.iteration += 1
        quiet || report(a, particles, schedule[end], d, false)
        if particles.llest <= llBound
            particles.reachedBound = true
            break
        end
        if particles.iteration < nits
            ##These steps can be skipped on the last iteration
            resample!(particles, acc)
            updateParticles!(a, particles, parameters)
        end
    end
    particles
end

##Adaptive version:
function doSMC(a::ApplicationDetails, parameters::Array{Float64, 1}, targetε::Float64; nparticles=200, k=ceil(Int, nparticles/2), llBound=-Inf, d=0, quiet=false)
    ##Initialise SMC by doing first iteration
    particles::Particles = Particles(a, nparticles, parameters)
    particles.ε = Inf
    particles.llest = lgamma(1+d/2)-(d/2)*log(pi)-d*log(targetε) ##log(1/V) where V is Lebesgue measure of a radius targetε and dimension d
    particles.iteration = 1 ##Number of *completed* iterations
    schedule = Float64[Inf]
    quiet || report(a, particles, targetε, d, true)

    ##Main loop
    while particles.ε > targetε
        dists::Array{Float64, 1} = getDistances(a, particles)
        particles.ε = max(targetε, select(dists, k))
        push!(schedule, particles.ε)
        acc = (dists .<= particles.ε)
        particles.llest += log(mean(acc))
        particles.iteration += 1
        quiet || report(a, particles, targetε, d, true)
        if (particles.llest <= llBound)
            particles.reachedBound = true
            break
        end
        if particles.ε == targetε
            break
        end
        resample!(particles, acc)
        updateParticles!(a, particles, parameters)
    end
    particles.schedule = schedule
    particles
end

##DEFAULT SMC FUNCTIONS

"
Perform a resampling step.
"
function resample!(particles::Particles, acc::BitArray)
    if sum(acc) == 0
        error("Resampling failed: no particles accepted")
    end
    N = particles.nparticles
    ind_alive = (1:N)[acc] ##Indices of alive particles
    ##Multinomial resampling
    indices = rand(ind_alive, N)

    ##Stratified resampling
    ##nAlive = sum(acc) ##Number of alive particles
    ##u = (1:N) - rand(N) ##Stratified uniform draws
    ##i = ceil(Int, u * nAlive/N) ##Random indices in 1:nAlive
    ##indices = ind_alive[i]

    ##Update the particles
    shuffle!(particles, indices)
    nothing
end

"
Report details of current iteration
"
function report(a::ApplicationDetails, p::Particles, targetε::Float64, d::Int, adaptive::Bool)
    print("Iteration $(p.iteration), eps=$(p.ε)")
    if !adaptive
        llest = p.llest
        if d>0 ##Avoids evaluating 0*Inf
            llest += d*(log(targetε) - log(p.ε))
        end
        print(", log of likelihood estimate=$llest, upper bound on final log likelihood estimate=$(p.llest)")
    end
    print("\n")
    nothing
end
