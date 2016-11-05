"
A multivariate distribution made up of independent component distributions.
"
immutable MvIndependent <: ContinuousMultivariateDistribution
    components::Array{ContinuousUnivariateDistribution, 1}
end

length(d::MvIndependent) = length(d.components)
_logpdf{T<:Real}(d::MvIndependent, x::AbstractVector{T}) = sum(Float64[logpdf(d.components[i], x[i]) for i in 1:length(d)])
function _rand!{T<:Real}(d::MvIndependent, x::AbstractVector{T})
    x = T[rand(c) for c in d.components]
end

