"
Maps `u` to an element (or array of elements) of [0,1]s by reflection.
"
function reflect(u::Float64)
    v = mod(u, 2.0)
    v > 1.0 ? 2.0-v : v
end

function reflect(u::Array{Float64, 1})
    Float64[reflect(x) for x in u]
end

"
Perform an iteration of slice sampling as in Murray and Graham.

## Arguments

- `u` Initial state vector of uniform variables
- `dosim` Function mapping a vector of uniforms to `sim, dist`, a simulation and a distance to the observations
- `epsilon` Acceptance threshold
- `w` Controls initial stepsize distribution

n.b. It's assumed that `dosim(u)[2] <= epsilon`.

## Returns

- A tuple `u, sim, z`: a new state vector, the corresponding simulation and the final stepsize.
"
function slicesample(u::Array{Float64, 1}, dosim::Function, epsilon::Float64; w=1.0)
    ##Initialisation
    p = length(u)
    v = randn(p)
    a = -rand()*w
    b = w + a
    ##Main loop
    while true
        z = a + rand() * (b-a)
        uprop = reflect(u+z*v)
        sim, dist = dosim(uprop) ##Profiling suggests this line doesn't cause type-stability problems
        if dist <= epsilon
            return uprop, sim, z
        end
        if z < 0
            a = z
        else
            b = z
        end
    end
end
