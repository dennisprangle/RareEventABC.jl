"
An iid mean zero normal dataset.

##Fields

- `yobs` The observed data.
- `nobs` Number of observations.
"
type IIDNormalDetails <: ApplicationDetails
    yobs::Array{Float64, 1}
    nobs::Int
end

"
A `IIDNormalParticles` variable stores the particles used in an SMC algorithm for the iid normal application.

##Fields

- `nparticles` How many particles are used.
- `scale` The scale parameter (must be >=0).
- `precision` The precision parameter i.e. `scale^-2.0`
- `u` Underlying uniform draws. `u[i]` is a vector of draws for the ith particle, controlling `y[i]`.
- `y` The simulated observation. `y[i]` is the simulated data for the ith particle.
- `accrate` Acceptance rate in the most recent iteration (fixed to 1.0 here)
- `iteration` How many SMC iterations have been performed.
- `reachedBound` Whether the lower bound on log likelihood has been reached.
- `schedule` Schedule of epsilon values used (or, for an adaptive algorithm, schedule so far)
- `maxz` Maximum slice sampling step size in previous iteration
"
type IIDNormalParticles <: Particles
    nparticles::Int
    scale::Float64
    precision::Float64
    u::Array{Array{Float64, 1}, 1}
    y::Array{Array{Float64, 1}, 1}
    accrate::Float64
    llest::Float64
    ε::Float64
    iteration::Int
    reachedBound::Bool
    schedule::Array{Float64, 1}
    maxz::Float64
end

"
Converts a vector of uniform draws to a vector of normal draws with the specified scale
"
function u2y(u::Array{Float64, 1}, scale::Float64)
    return quantile(Normal(), u) .* scale
end

function Particles(a::IIDNormalDetails, nparticles::Int, parameters::Array{Float64, 1})
    u = Array{Float64, 1}[rand(a.nobs) for i in 1:nparticles]
    y = Array{Float64, 1}[u2y(uvec, parameters[1]) for uvec in u]
    IIDNormalParticles(nparticles, parameters[1], parameters[1]^-2.0, u, y, 1.0, 0.0, Inf, 0, false, Float64[], 0.5)
end

"
Returns Euclidean distance between data vector `y` and observations.
"
function getDistance(a::IIDNormalDetails, y::Array{Float64, 1})
    sqrt(sum((y-a.yobs) .^ 2))
end

"
Returns vector of Euclidean distances between each particle and the observations.
"
function getDistances(a::IIDNormalDetails, p::IIDNormalParticles)
    Float64[getDistance(a, p.y[i]) for i in 1:p.nparticles]
end

"
Returns `NormalParticles` variable which is a copy of `p` except that its ith particle is a copy of the `indices[i]`th old particle.
"
function shuffle!(p::IIDNormalParticles, indices::Array{Int, 1})
    p.u = deepcopy(p.u[indices])
    p.y = deepcopy(p.y[indices])
    return nothing
end

"
Performs a slice sampling update on all particles.
"
function updateParticles!(a::IIDNormalDetails, p::IIDNormalParticles, θ::Array{Float64, 1})
    function dosim(u::Array{Float64, 1})
        y = u2y(u, θ[1])
        d = getDistance(a, y)
        return (y, d)
    end
    z = zeros(p.nparticles)
    for i in 1:p.nparticles
        p.u[i], p.y[i], z[i] = slicesample(p.u[i], dosim, p.ε, w=2.0*p.maxz)
    end
    p.maxz = maximum(abs(z))
    return nothing
end
