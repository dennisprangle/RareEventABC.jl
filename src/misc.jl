################################
##Miscellaneous useful functions
################################

"
Returns whether to accept a Metropolis-Hastings move with acceptance probability `min(1, exp(logα))`.
"
function accMH(logα::Float64)
    logα + rand(Distributions.Exponential()) > 0.0
end

"
Returns effective sample size for weights `w`, which do not need to be normalised.
"
function ess(w::Array{Float64, 1})
    if maximum(w)==0.0
        return 0.0
    end
    sum(w)^2 / sum(w.^2)
end

"
Inserts `x` into a sorted vector `v`. Sorting order is denoted by `rev`.
"
insertsorted!(v::Vector, x; rev=false) = insert!(v, searchsortedfirst(v,x,rev=rev), x)
