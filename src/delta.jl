"
A distribution which places all mass at value `x` of type `Float64`.

Only functions required elsewhere in the code are defined i.e. `rand`, `std`, `logpdf`.
Note `logpdf` is defined to return `1.0` or `-Inf` - this choice is to avoid `Inf-Inf` problems, rather than on a mathematical basis.
"
immutable DiracDelta <: ContinuousUnivariateDistribution
    x::Float64
end

std(d::DiracDelta) = 0.0
rand(d::DiracDelta) = d.x
logpdf(d::DiracDelta, x::Float64) = d.x == x ? 1.0 : -Inf
quantile(d::DiracDelta, x::Real) = d.x
quantile(d::DiracDelta, x::AbstractArray) = fill(d.x, size(x))
