module CW

import Classifier: predict!, update!, step!
import Base.zero

export CWParams
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


# Confidence Weighted
type CWParams{F,L}
    μW::Dict{F,Vector{Float64}}
    σW::Dict{F,Vector{Float64}}
    μb::Vector{Float64}
    σb::Vector{Float64}
    fwdlookup::Dict{L,Int}
    revlookup::Dict{Int,L}
end

function CWParams{L}(labels::Vector{L}, featuretype::DataType)
    μW = (featuretype=>Vector{Float64})[]
    σW = (featuretype=>Vector{Float64})[]
    μb = zeros(length(labels))
    σb = ones(length(labels))
    fwdlookup = [l=>i for (i,l) in enumerate(labels)]::Dict{L,Int}
    revlookup = [i=>l for (l,i) in fwdlookup]::Dict{Int,L}
    CWParams(μW, σW, μb, σb, fwdlookup, revlookup)
end

function outputs{F,L}(θ::CWParams{F,L}, x::Vector{F})
    ŷ = copy(θ.μb)
    for i in x
        if haskey(θ.μW, i)
            ŷ += θ.μW[i]
        end
    end
    ŷ
end

predict{F,L}(θ::CWParams{F,L}, x::Vector{F}) = indmax(outputs(θ, x)) 

const √2 = sqrt(2.0)
invcumnorm(x) = √2 * erfinv(2 * x - 1)

function alpha(m, v, p)
    term1 = 1.0 + 2.0 * p * m
    term2 = 8.0 * p * (m - p * v)
    denom = 4.0 * p * v
    (-term1 + sqrt(term1 * term1 - term2)) / denom
end

function update!{F,L}(θ::CWParams{F,L}, 
                      x::Vector{F}, 
                      y::Int, 
                      ŷout::Vector{Float64},
                      η::Float64)
    ŷ = indmax(ŷout)
    if ŷ == y
        ŷ = indmax2(ŷout, y)
    end
    
    m = ŷout[y] - ŷout[ŷ]

    v = θ.σb[y] + θ.σb[ŷ]
    for i in x
        σWi = get!(()->ones(length(θ.μb)), θ.σW, i)
        v += σWi[y]
        v += σWi[ŷ]
    end
    a = alpha(m, v, invcumnorm(η))     
    if a > 0 
        θ.μb[y] += a * θ.σb[y]
        θ.μb[ŷ] -= a * θ.σb[ŷ]
        θ.σb[y] = inv(inv(θ.σb[y]) + 2 * a)
        θ.σb[ŷ] = inv(inv(θ.σb[ŷ]) + 2 * a)
        for i in x
            μWi = get!(()->zero(θ.μb), θ.μW, i)
            σWi = θ.σW[i] # misses would have been caught above
            μWi[y] += a * σWi[y]
            μWi[ŷ] -= a * σWi[ŷ]
            σWi[y] = inv(inv(σWi[y]) + 2 * a)
            σWi[ŷ] = inv(inv(σWi[ŷ]) + 2 * a)
        end
    end
end

function step!{F,L}(θ::CWParams{F,L}, 
                    x::Vector{F}, 
                    ylabel::L, 
                    η::Float64)
    ŷout = outputs(θ, x)
    ŷ = indmax(ŷout)
    y = θ.fwdlookup[ylabel]
    update!(θ, x, y, ŷout, η)
    (y == ŷ) ? 0.0 : 1.0
end

end # module
