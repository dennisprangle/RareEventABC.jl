"
Defines a SEIR epidemic model and dataset.

##Fields

- `yobs` Vector of observed recovery times for each population member.
- `popsize` Number of population members.
- `get_pDist` A function `f(parameters::Array{Float64, 1})` returning a `Distribution` for pressure thresholds
- `get_eDist` A function `f(parameters::Array{Float64, 1})` returning a `Distribution` on time spent in the exposed stage
- `get_iDist` A function `f(parameters::Array{Float64, 1})` returning a `Distribution` on time spent in the infectious stage
- `round_times` A function `f(x::Float64)` mapping a removal time to rounded `Float64` time which will be used in the distance function. This is the identity mapping by default.
"
type SEIRDetails <: ApplicationDetails
    yobs::Array{Float64, 1}
    popsize::Int
    get_pDist::Function
    get_eDist::Function
    get_iDist::Function
    round_times::Function
end

function SEIRDetails(yobs::Array{Float64, 1}, popsize::Int, get_pDist::Function, get_eDist::Function, get_iDist::Function)
    SEIRDetails(yobs, popsize, get_pDist, get_eDist, get_iDist, x->x)
end

"
Summary of a single epidemic, including latent variables.

##Fields

- `popsize` Size of the population
- `pDist` Distribution of pressure thresholds
- `eDist` Distribution of time spent in the exposed stage
- `iDist` Distribution of time spent in the infectious stage
- `compartmentTimes` Element `[i,j]` is when individual i leaves compartment j
- `u_pressures` Uniform draws associated with `pressure` (see below). n.b. This is 1 shorter than pressures since the first pressure is fixed at 0.
- `u_eDuration` Uniform draws associated with `eDuration` (see below)
- `u_iDuration` Uniform draws associated with `iDuration` (see below)
- `pressures` Infectious pressures associated with each individual
- `eDuration` Duration in the E state (or hypothetical duration if that state not reached)
- `iDuration` Duration in the I state (or hypothetical duration if that state not reached)
- `nInfectiousAtExposure` How many individuals are infectious at this individual's exposure time (or zero if this individual never exposed)
- `totalPressure' Infectious pressure at end of epidemic
- `loglikelihood` The log likelihood of this epidemic
"
type EpidemicSummary
    popsize::Int
    pDist::Distributions.ContinuousUnivariateDistribution
    eDist::Distributions.ContinuousUnivariateDistribution
    iDist::Distributions.ContinuousUnivariateDistribution
    compartmentTimes::Array{Float64, 2}
    u_pressures::Array{Float64, 1}
    u_eDuration::Array{Float64, 1}
    u_iDuration::Array{Float64, 1}
    pressures::Array{Float64, 1}
    eDuration::Array{Float64, 1}
    iDuration::Array{Float64, 1}
    nInfectiousAtExposure::Array{Int, 1}
    totalPressure::Float64
    loglikelihood::Float64
end

"
Plot details of an epidemic `e`.
For SIR models, `hideExposed=true` omits details of the exposed stage.
"
function plotEpidemic(e::EpidemicSummary; hideExposed=false)
    timeMatrix = e.compartmentTimes
    whichInfected = !isinf(timeMatrix[:,3])
    eTimes = timeMatrix[whichInfected, 1]
    iTimes = timeMatrix[whichInfected, 2]
    rTimes = timeMatrix[whichInfected, 3]
    times = sort([eTimes; iTimes; rTimes])
    nS = map(t -> (e.popsize - sum(eTimes.<=t)), times)
    nE = map(t -> (sum(eTimes.<=t) - sum(iTimes.<=t)), times)
    nI = map(t -> (sum(iTimes.<=t) - sum(rTimes.<=t)), times)
    nR = map(t -> sum(rTimes.<=t), times)
    d = vcat(DataFrame(Time = times, Count=nS, Stage="Susceptible"),
             DataFrame(Time = times, Count=nI, Stage="Infectious"),
             DataFrame(Time = times, Count=nR, Stage="Recovered"))
    if !hideExposed
        d = vcat(d, DataFrame(Time = times, Count=nI, Stage="Exposed"))
    end
    plot(d, x=:Time, y=:Count, colour=:Stage, Geom.step)
end

"
A `SEIRParticles` variable stores the particles used in an SMC algorithm for a SEIR application.

##Fields

- `nparticles` How many particles are used
- `epidemics` Array of `EpidemicSummary` objects for each particle
- `llest` Log of likelihood estimate
- `ε` The current acceptance threshold
- `iteration` How many SMC iterations have been performed
- `reachedBound` Whether the lower bound on log likelihood has been reached
- `schedule` Schedule of epsilon values used (or, for an adaptive algorithm, schedule so far)
- `maxz` Maximum slice sampling step size in previous iteration
"
type SEIRParticles <: Particles
    nparticles::Int
    epidemics::Array{EpidemicSummary, 1}
    llest::Float64
    ε::Float64
    iteration::Int
    reachedBound::Bool
    schedule::Array{Float64, 1}
    maxz::Float64
end

function Particles(a::SEIRDetails, nparticles::Int, parameters::Array{Float64, 1})    
    epidemics = EpidemicSummary[simulateEpidemic(a, parameters) for i in 1:nparticles]
    SEIRParticles(nparticles, epidemics, 0.0, Inf, 0, false, Float64[], 0.5)
end

"
Return an `EpidemicSummary` variable given model details `a` and parameters `θ`.
"
function simulateEpidemic(a::SEIRDetails, θ::Array{Float64, 1})
    λscaled = θ[1]/a.popsize
    u_pressures = rand(a.popsize-1)
    u_eDuration = rand(a.popsize)
    u_iDuration = rand(a.popsize)
    pDist::Distributions.ContinuousUnivariateDistribution = a.get_pDist(θ)
    eDist::Distributions.ContinuousUnivariateDistribution = a.get_eDist(θ)
    iDist::Distributions.ContinuousUnivariateDistribution = a.get_iDist(θ)
    makeEpidemic(λscaled, u_pressures, u_eDuration, u_iDuration, pDist, eDist, iDist)
end

"
Return an `EpidemicSummary` variable corresponding to exposure parameter `λscaled`, and specified `pressureThresholds`, `eDuration` and `iDuration` arrays.
The `pDist`, `eDist` and `iDist` distributions must also be supplied.
Note it is assumed that `pressureThresholds[1]==0.0` i.e. individual 1 is exposed at time zero.
"
function makeEpidemic(λscaled::Float64, u_pressures::Array{Float64, 1}, u_eDuration::Array{Float64, 1}, u_iDuration::Array{Float64, 1}, pDist::Distributions.ContinuousUnivariateDistribution, eDist::Distributions.ContinuousUnivariateDistribution, iDist::Distributions.ContinuousUnivariateDistribution)
    ##Initialisation
    pressureThresholds = zeros(1+length(u_pressures))
    pressureThresholds[2:end] = quantile(pDist, u_pressures) ##nb pressureThresholds[1] is fixed at 0.0 as it's the initial infective
    eDuration = quantile(eDist, u_eDuration)
    iDuration = quantile(iDist, u_iDuration)
    popsize = length(pressureThresholds)
    compartmentTimes = fill(Inf, (popsize, 3))
    nInfectiousAtExposure = [1; zeros(Int, popsize-1)]
    compartmentTimes[1, :] = [0.0, eDuration[1], eDuration[1]+iDuration[1]]
    orderedFutureEIs = Float64[compartmentTimes[1,2]]
    orderedFutureIRs = Float64[compartmentTimes[1,3]]
    orderedThresholds = sort(pressureThresholds[2:end], rev=true)
    futureSEIndices = sortperm(pressureThresholds, rev=true)
    pop!(futureSEIndices) ##Get rid of initial infective
    t = 0.0
    pressure = 0.0
    nInfectious = 0
    if length(orderedThresholds) > 0
        nextThreshold = pop!(orderedThresholds)
        nextExposed = pop!(futureSEIndices)
    else
        nextThreshold = Inf ##Don't update nextExposed as value irrelevant
    end

    ##Main loop
    while length(orderedFutureEIs)+length(orderedFutureIRs)>0
        nextEITime = length(orderedFutureEIs) > 0 ? orderedFutureEIs[end] : Inf
        nextIRTime = length(orderedFutureIRs) > 0 ? orderedFutureIRs[end] : Inf
        nextEventTime = min(nextEITime, nextIRTime)
        nextEventType = nextEITime < nextIRTime ? 1 : -1 ##Change in number infectious
        nextPressure = pressure + (nextEventTime - t)*nInfectious*λscaled
        if nextThreshold < nextPressure
            timeExposed = t + (nextThreshold - pressure)/(nInfectious*λscaled)
            timeInfectious = timeExposed + eDuration[nextExposed]
            timeRecovered = timeInfectious + iDuration[nextExposed]
            compartmentTimes[nextExposed, :] = [timeExposed, timeInfectious, timeRecovered]
            insertsorted!(orderedFutureEIs, timeInfectious, rev=true)
            insertsorted!(orderedFutureIRs, timeRecovered, rev=true)
            nInfectiousAtExposure[nextExposed] = nInfectious
            pressure = nextThreshold
            if length(orderedThresholds) > 0
                nextThreshold = pop!(orderedThresholds)
                nextExposed = pop!(futureSEIndices)
            else
              nextThreshold = Inf ##Don't update nextExposed as value irrelevant
            end
            t = timeExposed
        else
            pressure = nextPressure
            t = nextEventTime
            nInfectious += nextEventType
            if nextEventType == 1
                pop!(orderedFutureEIs)
            else
                pop!(orderedFutureIRs)
            end
        end
    end
    e = EpidemicSummary(popsize, pDist, eDist, iDist, compartmentTimes, u_pressures, u_eDuration, u_iDuration, pressureThresholds, eDuration, iDuration, nInfectiousAtExposure, pressure, 0.0)
    e.loglikelihood = logLikelihood(e)
    return e
end

"
Returns `SEIRParticles` variable which is a copy of `p` except that its ith particle is a copy of the `indices[i]`th old particle.
"
function shuffle!(p::SEIRParticles, indices::Array{Int, 1})
    epidemics_new = Array(EpidemicSummary, p.nparticles)
    for i in 1:p.nparticles
        epidemics_new[i] = deepcopy(p.epidemics[indices[i]])
    end
    p.epidemics[:] = epidemics_new[:]
    return nothing
end

"
Returns distance between observation time vector `y` and observations specified in `a`.

The distance is the sum of two components.
The first is Euclidean distance between sorted values, excluding any which are infinite in either vector after sorting.
The second is the difference in the number of finite values multiplied by `penalty`.
`penalty` should be chosen so that the second component is much larger.
"
function getDistance(a::SEIRDetails, e::EpidemicSummary; penalty=1000.0)
    n = length(a.yobs)
    robs = sort(a.yobs) ##robs = observed removal times
    rsim = sort(e.compartmentTimes[:,3]) ##rsim = simulated removal times
    psim = sort(e.pressures) ##psim = observed pressure thresholds
    ##Convert to inter-removal times
    if robs[1] < Inf
        robs .-= robs[1]
    end
    if rsim[1] < Inf
        rsim .-= rsim[1]
    end
    ##Find sum square difference of rounded removal times, where both are finite
    nFinite_obs = sum(isfinite(robs))
    nFinite_sim = sum(isfinite(rsim))
    nFinite_both = min(nFinite_obs, nFinite_sim)
    robs_round = map(a.round_times, robs[1:nFinite_both])
    rsim_round = map(a.round_times, rsim[1:nFinite_both])
    dist = sqrt(sum((robs_round .- rsim_round) .^ 2))
    ##Penalise for different numbers of removals
    dist += penalty*abs(nFinite_obs - nFinite_sim)
    ##Penalise for pressures
    if (nFinite_sim > nFinite_obs)
        dist += e.totalPressure*(nFinite_sim-nFinite_obs) - sum(psim[(nFinite_obs+1):nFinite_sim])
    else
        dist += sum(psim[(nFinite_sim+1):nFinite_obs])
    end
    return dist
end

"
Returns vector of Euclidean distances between each particle and the observations.
"
function getDistances(a::SEIRDetails, p::SEIRParticles)
    Float64[getDistance(a, e) for e in p.epidemics]
end

"
Performs an MCMC update on all particles, using slice sampling.

The new pressures and durations are inputed to the `makeEpidemic` function to derive the resulting event times.
"
function updateParticles!(a::SEIRDetails, p::SEIRParticles, θ::Array{Float64, 1})
    n = a.popsize
    λscaled = θ[1]/n
    pDist::Distributions.ContinuousUnivariateDistribution = p.epidemics[1].pDist
    eDist::Distributions.ContinuousUnivariateDistribution = p.epidemics[1].eDist
    iDist::Distributions.ContinuousUnivariateDistribution = p.epidemics[1].iDist
    function dosim(u::Array{Float64, 1})
        u_pressures = u[1:(n-1)]
        u_eDuration = u[n:(2*n-1)]
        u_iDuration = u[(2*n):(3*n-1)]
        e = makeEpidemic(λscaled, u_pressures, u_eDuration, u_iDuration, pDist, eDist, iDist)
        return (e, getDistance(a, e))
    end
    z = zeros(p.nparticles)
    for i in 1:p.nparticles
        ecurr = p.epidemics[i]
        newu, p.epidemics[i], z[i] = slicesample([ecurr.u_pressures; ecurr.u_eDuration; ecurr.u_iDuration], dosim, p.ε, w=2.0*p.maxz)
    end
    p.maxz = maximum(abs(z))
    return nothing
end
   
"
Return log likelihood of epidemic `e`
"
function logLikelihood(e::EpidemicSummary)
    all(e.pressures .>= 0.0) || return(-Inf)
    ll = 0.0
    for i in 1:e.popsize
        ll += logpdf(e.pDist, e.pressures[i])
        ll += logpdf(e.eDist, e.eDuration[i])
        ll += logpdf(e.iDist, e.iDuration[i])
    end
    return ll
end
