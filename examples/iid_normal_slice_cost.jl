##INVESTIGATE COST OF SLICE SAMPLING WITHIN RE-ABC
using RareEventABC

##GENERATE OBSERVATIONS
srand(1);
d = 25;
yobs = randn(d) .* 3.0;
a = RareEventABC.IIDNormalDetails(yobs, d);

##INITIALISE SMC TUNING CHOICES
nparticles = 200;
parameters = [3.0];
schedule = doSMC(a, [3.0], 1.35, nparticles=200, quiet=true).schedule

##SPECIALISED VERSIONS OF SMC SUBROUTINES FOR THIS INVESTIGATION
##Slice sampling outputting the number of iterations required
function slicesample_(u::Array{Float64, 1}, dosim::Function, epsilon::Float64; w=1.0)
    ##Initialisation
    p = length(u)
    v = randn(p)
    a = -rand()*w
    b = w + a
    nits = 0
    ##Main loop
    while true
        z = a + rand() * (b-a)
        uprop = RareEventABC.reflect(u+z*v)
        sim, dist = dosim(uprop) ##Profiling suggests this line doesn't cause type-stability problems
        nits += 1
        if dist <= epsilon
            return uprop, sim, z, nits
        end
        if z < 0
            a = z
        else
            b = z
        end
    end
end

##Update particles storing mean number of slice sampling iterations required
function updateParticles_!(a::RareEventABC.IIDNormalDetails, p::RareEventABC.IIDNormalParticles, θ::Array{Float64, 1}, mean_ss_its::Vector; adaptive_w = true)
    function dosim(u::Array{Float64, 1})
        y = RareEventABC.u2y(u, θ[1])
        d = RareEventABC.getDistance(a, y)
        return (y, d)
    end
    z = zeros(p.nparticles)
    s = zeros(p.nparticles)
    for i in 1:p.nparticles
        if adaptive_w
            p.u[i], p.y[i], z[i], s[i] = slicesample_(p.u[i], dosim, p.ε, w=min(1.0,2.0*p.maxz))
        else
            p.u[i], p.y[i], z[i], s[i] = slicesample_(p.u[i], dosim, p.ε)
        end
    end    
    p.maxz = maximum(abs(z))
    push!(mean_ss_its, mean(s))
    return nothing
end

##SMC - as usual but call functions above
function doSMC_(a::RareEventABC.ApplicationDetails, parameters::Array{Float64, 1}, schedule::Array{Float64, 1}, mean_ss_its::Vector; nparticles=200, llBound=-Inf, d=0, quiet=false, adaptive_w = true)
    ##Initialise SMC, including sampling from prior
    particles::RareEventABC.Particles = RareEventABC.Particles(a, nparticles, parameters)
    particles.schedule = schedule
    particles.llest = lgamma(1+d/2)-(d/2)*log(pi)-d*log(schedule[end]) ##log(1/V) where V is Lebesgue measure of a radius schedule[end] and dimension d
    particles.iteration = 0 ##Number of *completed* iterations
    particles.maxz = 1.0 ##Override what's currently in RareEventABC!

    ##Main loop
    nits = length(schedule)
    for ε in schedule
        particles.ε = ε
        dists::Array{Float64, 1} = RareEventABC.getDistances(a, particles)
        acc = (dists .<= ε)
        particles.llest += log(mean(acc))
        particles.iteration += 1
        ##quiet || report(a, particles, schedule[end], d, false)
        if particles.llest <= llBound
            particles.reachedBound = true
            break
        end
        if particles.iteration < nits
            ##These steps can be skipped on the last iteration
            RareEventABC.resample!(particles, acc)
            updateParticles_!(a, particles, parameters, mean_ss_its, adaptive_w=adaptive_w)
        end
    end
    particles
end

##A DETAILED COMPARISON
S_adaptive = Float64[];
S_nonadaptive = Float64[];
srand(2);
doSMC_(a, [3.0], schedule, S_adaptive, nparticles=200, quiet=true, adaptive_w = true).llest
doSMC_(a, [3.0], schedule, S_nonadaptive, nparticles=200, quiet=true, adaptive_w = false).llest

using DataFrames
using Plots; pgfplots()
using StatPlots
df = vcat(DataFrame(iteration=1:length(schedule)-1, mean_its=S_adaptive, w="adaptive"),
          DataFrame(iteration=1:length(schedule)-1, mean_its=S_nonadaptive, w="nonadaptive"));
default(guidefont=font(25), tickfont=font(25), legendfont=font(25), size=(500,500));
p1 = scatter(df, :iteration, :mean_its, group=:w, marker=:auto,
             xlabel="SMC iteration",
             ylabel="Mean slice sampling iterations",
             ylims=(1.0,11.0)); ##Hack to prevent legend getting in the way
savefig(p1, "slice_cost.pdf")

##REPEATED COMPARISONS TO SEE HOW TIME COST AND LIKELIHOOD ESTIMATES DIFFER
srand(0);
ll_adaptive = @time Float64[doSMC_(a, [3.0], schedule, S_adaptive, nparticles=200, quiet=true, adaptive_w = true).llest for i in 1:50];
ll_nonadaptive = @time Float64[doSMC_(a, [3.0], schedule, S_nonadaptive, nparticles=200, quiet=true, adaptive_w = false).llest for i in 1:50];

using HypothesisTests
UnequalVarianceTTest(exp(ll_adaptive), exp(ll_nonadaptive))

std(ll_adaptive)
std(ll_nonadaptive)

    
