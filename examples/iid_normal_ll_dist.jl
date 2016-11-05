##INVESTIGATE DISTRIBUTION OF RE-ABC LIKELIHOOD ESTIMATES
using RareEventABC
using ProgressMeter
using DataFrames
using Distributions
using Plots; pgfplots()
using StatPlots

##GENERATE OBSERVATIONS
srand(1);
d = 25;
yobs = randn(d) .* 3.0;
a = RareEventABC.IIDNormalDetails(yobs, d);

##GENERATE LIKELIHOOD ESTIMATES
schedule = doSMC(a, [3.0], 5.0, nparticles=200, quiet=true).schedule
llexperiment = Dict();
@showprogress for nparticles in 10:10:100
    llest = Float64[doSMC(a, [3.0], schedule, nparticles=nparticles, d=25, quiet=true).llest for i in 1:100]
    llexperiment[nparticles] = llest
end

##QQ PLOTS
theoretical_quantiles = quantile(Normal(), (1:100)/101)
qqdf = DataFrame(samp=Float64[], theoretical=Float64[], nparticles=Int[])
for nparticles in 20:10:50
    ll = llexperiment[nparticles]
    temp = DataFrame(samp=sort(ll), theoretical=theoretical_quantiles, particles=nparticles)
    qqdf = vcat(qqdf, temp)
end

qqdf2 = qqdf[qqdf[:samp] .> -Inf, :] ##Plots.jl doesn't seem to work with infinite values
default(guidefont=font(25), tickfont=font(25), legendfont=font(25), size=(500,500));
p1 = scatter(qqdf2, :theoretical, :samp, group=:particles,
             markershape=:auto,
             xlims=(-2.5,3.0), ##Hack so that legend isn't in the way
             xlabel="Sample quantile", ylabel="Theoretical quantile");
savefig(p1, "iidnormal_qq.pdf")

    
##COMPARE ADAPTIVE AND NON-ADAPTIVE ALGORITHMS
b = RareEventABC.IIDNormalDetails([0.0], 1);
eps = 0.2;
##trueL = (cdf(n, eps) - cdf(n,-eps)) / (2*eps)

nparticles = 10;
    
srand(1);
nreps = 10000;
likelihood_adaptive = Float64[doSMC(b, [1.0], eps, nparticles=nparticles, d=1, quiet=true).llest for i in 1:nreps] |> exp;

srand(2);
schedule = doSMC(b, [1.0], eps, nparticles=nparticles, quiet=true).schedule
likelihood_nonadaptive = Float64[doSMC(b, [1.0], schedule, nparticles=nparticles, d=1, quiet=true).llest for i in 1:nreps] |> exp;
    
mean(likelihood_adaptive), std(likelihood_adaptive)/sqrt(nreps)
mean(likelihood_nonadaptive), std(likelihood_nonadaptive)/sqrt(nreps)
