##USED TO CREATE IID NORMAL FIGURE FOR ABC
using RareEventABC
using Distributions
using DataFrames
using ProgressMeter
using Lora
using Plots; pgfplots();
using StatPlots;

##Generate observations
srand(1);
d = 25;
yobs = randn(d) .* 3.0;
a = RareEventABC.IIDNormalDetails(yobs, d);

rmse(x::Vector, y::Float64) = (x .- y) .^ 2 |> mean |> sqrt;
rmse(x::Vector, y::Float64, w::Vector) = (w .* ((x .- y) .^ 2)) |> sum |> a->a/sum(w) |> sqrt;

######
###ABC
######

"
Run a trial of ABC

Arguments:

* `N` is number of acceptances required
* `yobs` is observed data
* `threshold` is threshold to investigate
* `scale_true` is the true parameter value
"
function trialABC(N::Int, yobs::Vector, threshold::Float64; scale_true=3.0)
    prior = Uniform(0.0,10.0)
    output_sample = Float64[]
    nsims = 0
    tic()
    @showprogress for i in 1:N
        while true
            prior_draw = rand(prior)
            next_dist = RareEventABC.getDistance(a, randn(d) .* prior_draw)
            nsims += 1
            if next_dist <= threshold
                push!(output_sample, prior_draw)
                break ##i.e. increment i in for loop
            end ##if
        end ##while
    end ##for
    timeABC = toq()
    error = rmse(output_sample, scale_true)
    time_per_sample = timeABC / N
    return DataFrame(threshold = threshold, time_per_ESS = time_per_sample, RMSE = error, accrate = N/nsims, method="ABC", dim=d, mean = mean(output_sample), sd = std(output_sample))
end ##trialABC

srand(2);
temp = [trialABC(500, yobs, eps) for eps in 9.0:30.0];
abc_efficiency = vcat(temp...)

#######
##RE-ABC (non-adaptive)
#######

##Tuning
thresholds_hd = [5.0, 10.0, 15.0, 20.0, 25.0];
particles_hd  = [300, 40,  16,   10,    10];

srand(1);
schedules_hd = Array{Float64, 1}[doSMC(a, [3.0], t, nparticles=p, quiet=true).schedule for (t,p) in zip(thresholds_hd, particles_hd)];
llest = Float64[doSMC(a, [3.0], s, nparticles=p, quiet=true).llest for (s,p) in zip(schedules_hd, particles_hd), i in 1:100];
var(llest, 2)
fin(x) = x[x .> -Inf]; ##Returns finite elements
[var(fin(llest[i,:])) for i in 1:size(llest)[1]] ##Variance if zeros likelihood estimates excluded

"
Run a trial of RE-ABC within PMMH

Arguments:

* `nits` is number of PMMH iterations required
* `yobs` is observed data
* `schedule` is threshold schedule (non-adaptive algorithm)
* `epsilon` is final threshold (adaptive algorithm)
* `scale_true` is the true parameter value
* `scale_init` is starting MCMC state
* `nparticles` is number of SMC particles
* `sd` is standard deviation of MH proposal distribution
"
function trialREABC(nits::Int, yobs::Vector, schedule::Array{Float64, 1}; scale_true=3.0, scale_init=3.0, nparticles=1000, sd=1.0)
    tic()
    thetas, ll = doPMMH(a, schedule, nits, nparticles, Uniform(0.0,10.0), scale_init, Normal(0.0,sd));
    t = toq()
    return DataFrame(threshold = schedule[end], time_per_ESS = t/Lora.ess(thetas), RMSE = rmse(thetas[:,1], scale_true), accrate = length(unique(thetas)) / nits, method="RE-ABC", dim=d, mean = mean(thetas), sd = std(thetas))
end

##Pilot run
srand(2);
sd_hd = sqrt(2.562) * Float64[trialREABC(200, yobs, sched, scale_init=3.0, nparticles=p, sd=0.5)[1,:sd] for (sched,p) in zip(schedules_hd, particles_hd)]

##Main run
srand(3);
temp = [trialREABC(2000, yobs, sched, scale_init=3.0, nparticles=p, sd=s) for (sched,p,s) in zip(schedules_hd, particles_hd, sd_hd)];
abc_hd_efficiency = vcat(temp...)

##Repeat with "bad" schedules
particles_bad  = [300, 150,  60,   20,    10];
srand(1);
schedules_bad = Array{Float64, 1}[doSMC(a, [6.0], t, nparticles=p, quiet=true).schedule for (t,p) in zip(thresholds_hd, particles_bad)];
llest = @showprogress Float64[doSMC(a, [6.0], s, nparticles=p, quiet=true).llest for (s,p) in zip(schedules_bad, particles_bad), i in 1:100];
var(llest, 2)
fin(x) = x[x .> -Inf]; ##Returns finite elements
[var(fin(llest[i,:])) for i in 1:size(llest)[1]] ##Variance if zeros likelihood estimates excluded

##Pilot run
srand(2);
sd_bad = sqrt(2.562) * Float64[trialREABC(200, yobs, sched, scale_init=3.0, nparticles=p, sd=0.5)[1,:sd] for (sched,p) in zip(schedules_bad, particles_bad)]

##Main run
srand(3);
temp = [trialREABC(2000, yobs, sched, scale_init=3.0, nparticles=p, sd=s) for (sched,p,s) in zip(schedules_bad, particles_bad, sd_bad)];
efficiency_bad = vcat(temp...)
efficiency_bad[:method] = "RE-ABC poor schedule"
    
#######
##RE-ABC (adaptive)
#######

##Tuning
thresholds_hd = [5.0, 10.0, 15.0, 20.0, 25.0];
particles_ahd  = [260, 40,  10,   10,    10];

srand(1);
llest = Float64[doSMC(a, [3.0], t, nparticles=p, quiet=true).llest for (t,p) in zip(thresholds_hd, particles_ahd), i in 1:100];
var(llest, 2)
fin(x) = x[x .> -Inf]; ##Returns finite elements
[var(fin(llest[i,:])) for i in 1:size(llest)[1]] ##Variance if zeros likelihood estimates excluded

    
##Adaptive version
function trialREABC(nits::Int, yobs::Vector, epsilon::Float64; scale_true=3.0, scale_init=3.0, nparticles=1000, sd=1.0)
    tic()
    thetas, ll = doPMMH(a, epsilon, nits, nparticles, Uniform(0.0,10.0), scale_init, Normal(0.0,sd));
    t = toq()
    return DataFrame(threshold = epsilon, time_per_ESS = t/Lora.ess(thetas), RMSE = rmse(thetas[:,1], scale_true), accrate = length(unique(thetas)) / nits, method="Adaptive RE-ABC", dim=d, mean = mean(thetas), sd = std(thetas))
end

##Pilot run
srand(2);
sd_ahd = sqrt(2.562) * Float64[trialREABC(200, yobs, t, scale_init=3.0, nparticles=p, sd=0.5)[1,:sd] for (t,p) in zip(thresholds_hd, particles_ahd)]

##The results above are sufficiently similar to the non-adaptive case that we use the non-adaptive tuning for a clearer comparison
srand(3);
temp = [trialREABC(2000, yobs, t, scale_init=3.0, nparticles=p, sd=s) for (t,p,s) in zip(thresholds_hd, particles_hd, sd_hd)];
abc_ahd_efficiency = vcat(temp...)

##Closer comparison of threshold 5 case
thetas_a, ll_a = doPMMH(a, 5.0, 2000, 300, Uniform(0.0,10.0), 3.0, Normal(0.0,0.75));
thetas_n, ll_n = doPMMH(a, schedules_hd[1], 2000, 300, Uniform(0.0,10.0), 3.0, Normal(0.0,0.75));
plot(x=1:2000, hcat(thetas_a, thetas_n), line=:steppost, layout=2)
Lora.ess(thetas_a), Lora.ess(thetas_n)
    
#######
##LIKELIHOOD-BASED INFERENCE
#######
"
Likelihood based inference using importance sampling

Arguments:

* `nits` is number of iterations to perform
* `yobs` is observed data
* `scale_true` is the true parameter value
"
function likelihoodInf(nits::Int, yobs::Vector; scale_true=3.0)
    prior = Uniform(0.0,10.0)
    output_sample = Array(Float64, nits)
    w = Array(Float64, nits)
    tic()
    @showprogress for i in 1:nits
        sig = output_sample[i] = rand(prior)
        w[i] = pdf(Distributions.MvNormal(zeros(d), sig), yobs)
    end ##for
    timeIS = toq()
    error = rmse(output_sample, scale_true, w)
    return DataFrame(threshold = 0.0, time_per_ESS = timeIS / RareEventABC.ess(w), RMSE = error, accrate = 1.0, method="Exact", dim=d)
end ##trialABC

srand(3);
IS_efficiency = likelihoodInf(10000, yobs)


"
Likelihood based inference using MCMC

Arguments:

* `nits` is number of iterations to perform
* `yobs` is observed data
* `scale_true` is the true parameter value
"
function doMCMC(nits::Int, yobs::Vector; scale_true=3.0, scale_init=3.0, sd=1.0, thin=1)
    tic()
    thetas = Float64[]
    theta = scale_init
    L = pdf(Distributions.MvNormal(zeros(d), theta), yobs)
    1 % thin == 0 && push!(thetas, theta)
    tic()
    @showprogress for i in 2:nits
        theta_prop = theta + sd*randn()
        if 0.0 < theta_prop < 10.0
            L_prop = pdf(Distributions.MvNormal(zeros(d), theta_prop), yobs)
            if rand() < L_prop / L
                theta = theta_prop ##Acceptance
            end
        end
        i % thin == 0 && push!(thetas, theta)
    end
    t = toq()
    return DataFrame(threshold = 0.0, time_per_ESS = t/Lora.ess(thetas), RMSE = rmse(thetas[:,1], scale_true), accrate = length(unique(thetas)) / nits, method="MCMC", dim=d, mean = mean(thetas), sd = std(thetas))
end

srand(3);
MCMC_efficiency = doMCMC(10000, yobs, sd=1.8)

#######
##ABC-MCMC
#######
    
"
Run ABC-MCMC

Arguments:

* `nits` is number of iterations required
* `yobs` is observed data
* `threshold` is threshold to investigate
* `scale_true` is the true parameter value
* `scale_init` is starting MCMC state
* `sd` is standard deviation of MH proposal distribution
* `thin` is a thinning factor. This many iterations are performed for each one stored.
"
function doABCMCMC(nits::Int, yobs::Vector, threshold::Float64; scale_true=3.0, scale_init=3.0, sd=1.0, thin=1)
    tic()
    thetas = Float64[]
    theta = scale_init
    1 % thin == 0 && push!(thetas, theta)
    @showprogress for i in 2:nits
        theta_prop = theta + sd*randn()
        if 0.0 < theta_prop < 10.0
            next_dist = RareEventABC.getDistance(a, randn(d) .* theta_prop)
            if next_dist <= threshold
                theta = theta_prop ##Acceptance
            end
        end
        i % thin == 0 && push!(thetas, theta)
    end
    t = toq()
    return DataFrame(threshold = threshold, time_per_ESS = t/Lora.ess(thetas), RMSE = rmse(thetas[:,1], scale_true), accrate = length(unique(thetas)) / nits, method="ABC-MCMC", dim=d, mean = mean(thetas), sd = std(thetas))
end

prop_sds = sqrt(2.562) * abc_efficiency[:,:sd].data;
    
srand(3);
temp = [doABCMCMC(10^5, yobs, t, scale_init=3.0, sd=s) for (t,s) in zip(11.0:30.0, prop_sds[3:end])];
push!(temp, doABCMCMC(10^8, yobs, 10.0, scale_init=3.0, sd=prop_sds[2], thin=10^4));
push!(temp, doABCMCMC(2*10^8, yobs, 9.0, scale_init=3.0, sd=prop_sds[1], thin=10^4));
abc_mcmc_efficiency = vcat(temp...)
 
#######
##PLOTS
#######

eff = vcat(abc_efficiency, abc_hd_efficiency, abc_ahd_efficiency, MCMC_efficiency, abc_mcmc_efficiency);
##eff = vcat(eff, efficiency_bad);

sort!(eff);
default(guidefont=font(25), tickfont=font(25), legendfont=font(25), size=(500,500));
p1 = scatter(eff, :threshold, :RMSE, group=:method,
        markershape=:auto, line=:path);
savefig(p1, "iid_normal_accuracy.pdf") ##For talk
p1 = scatter!(p1, legend=false);
p2 = scatter(eff, :threshold, :time_per_ESS, group=:method,
        markershape=:auto, line=:path, yaxis=("time / ESS (s)", :log10),
        ylims=(1E-6,10.0));
savefig(p2, "iid_normal_efficiency.pdf") ##For talk
p3 = plot(p1,p2, size=(1000,500));
savefig(p3, "iid_normal.pdf") ##For paper

##For talk, just plot ABC results
##abc_efficiency = eff[eff[:method].=="ABC",:];
p4 = scatter(abc_efficiency,
             :threshold, :RMSE, 
             markershape=:auto, line=:path, ylims=(0.0,2.5), legend=false);
p5 = scatter(abc_efficiency,
             :threshold, :time_per_ESS, group=:method,
             markershape=:auto, line=:path, yaxis=("time / ESS (s)", :log10),
             ylims=(1E-6,10.0), legend=false);
p6 = plot(p4,p5, size=(1000,500));
savefig(p6, "iid_normal_abc_only.pdf")
