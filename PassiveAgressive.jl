module PassiveAgressive

import Classifier: predict!, update!, step!
import Base.zero

export PassiveAgressiveParams
export predict!, update!, step!

# given i = indmax(x) return 
# argmax{x[j] | i != j}
function indmax2{T}(x::Vector{T}, i::Int)
    maxj, maxval = 0, typemin(T)
    for j = 1:length(x)
        if i != j && x[j] > maxval
            maxj = j
            maxval = x[j]
        end
    end
    maxj
end

# Passive Agressive 
type PassiveAgressiveParams{F,L}
    W::Dict{F,Vector{Float64}}
    b::Vector{Float64}
    fwdlookup::Dict{L,Int}
    revlookup::Dict{Int,L}
end

function PassiveAgressiveParams{L}(labels::Vector{L}, featuretype::DataType)
    W = (featuretype=>Vector{Float64})[]
    b = zeros(length(labels))
    fwdlookup = [l=>i for (i,l) in enumerate(labels)]::Dict{L,Int}
    revlookup = [i=>l for (l,i) in fwdlookup]::Dict{Int,L}
    PassiveAgressiveParams(W, b, fwdlookup, revlookup)
end

function outputs{F,L}(θ::PassiveAgressiveParams{F,L}, x::Vector{F})
    ŷ = copy(θ.b)
    for i in x
        if haskey(θ.W, i)
            ŷ += θ.W[i]
        end
    end
    ŷ
end

predict{F,L}(θ::PassiveAgressiveParams{F,L}, x::Vector{F}) = indmax(outputs(θ, x)) 

function update!{F,L}(θ::PassiveAgressiveParams{F,L}, 
                      x::Vector{F}, 
                      y::Int, 
                      ŷout::Vector{Float64}, 
                      C::Float64,
                      variant::Symbol)
    ŷ = indmax(ŷout)
    if ŷ == y
        ŷ = indmax2(ŷout, y)
    end
    margin = ŷout[y] - ŷout[ŷ]
    if margin < 1.0
        lmc = 1.0 - margin
        xnorm = sqrt(2.0 * length(x))
        τ = 0.0
        if variant == :PA2
            τ = lmc / (xnorm + (1.0 / (2.0 * C))) # PA-II
        elseif variant == :PA1
            τ = min(C, lmc / xnorm) # PA-I
        else
            τ = lmc / xnorm # PA
        end
        θ.b[y] += τ
        θ.b[ŷ] -= τ
        for i in x
            Wi = get!(()->zero(θ.b), θ.W, i)
            Wi[y] += τ
            Wi[ŷ] -= τ
        end
    end
end

function step!{F,L}(θ::PassiveAgressiveParams{F,L}, 
                    x::Vector{F}, 
                    ylabel::L, 
                    C::Float64, 
                    variant::Symbol=:PA2)
    ŷout = outputs(θ, x)
    ŷ = indmax(ŷout)
    y = θ.fwdlookup[ylabel]
    update!(θ, x, y, ŷout, C, variant)
    (y == ŷ) ? 0.0 : 1.0
end

end # module
