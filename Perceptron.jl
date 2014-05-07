module Perceptron

import Classifier: predict!, update!, step!
import Base.zero

export PerceptronParams, AvgPerceptronParams
export predict!, update!, step!

# Perceptron
type PerceptronParams{F,L}
    W::Dict{F,Vector{Float64}}
    b::Vector{Float64}
    fwdlookup::Dict{L,Int}
    revlookup::Dict{Int,L}
end

function PerceptronParams{L}(labels::Vector{L}, featuretype::DataType)
    W = (featuretype=>Vector{Float64})[]
    b = zeros(length(labels))
    fwdlookup = [l=>i for (i,l) in enumerate(labels)]::Dict{L,Int}
    revlookup = [i=>l for (l,i) in fwdlookup]::Dict{Int,L}
    PerceptronParams(W, b, fwdlookup, revlookup)
end

function predict{F,L}(θ::PerceptronParams{F,L}, x::Vector{F})
    ŷ = copy(θ.b)
    for i in x
        if haskey(θ.W, i)
            ŷ += θ.W[i]
        end
    end
    indmax(ŷ) 
end

function update!{F,L}(θ::PerceptronParams{F,L}, x::Vector{F}, y::Int, ŷ::Int)
    θ.b[y] += 1.0
    θ.b[ŷ] -= 1.0
    for i in x
        Wi = get!(()->zero(θ.b), θ.W, i)
        Wi[y] += 1.0
        Wi[ŷ] -= 1.0
    end
end

function step!{F,L}(θ::PerceptronParams{F,L}, x::Vector{F}, ylabel::L)
    ŷ = predict(θ, x)
    y = θ.fwdlookup[ylabel]
    if y != ŷ
        update!(θ, x, y, ŷ)
        return 1.0
    else
        return 0.0
    end
end

# Averaged Perceptron
type ParamTriple
    value::Float64
    total::Float64
    stamp::Float64
end
ParamTriple() = ParamTriple(0.0, 0.0, 0.0)
zero(v::Vector{ParamTriple}) = [ParamTriple() for i = 1:length(v)]

type AvgPerceptronParams{F,L}
    W::Dict{F,Vector{ParamTriple}}
    b::Vector{ParamTriple}
    fwdlookup::Dict{L,Int}
    revlookup::Dict{Int,L}
    t::Float64
end

function AvgPerceptronParams{L}(labels::Vector{L}, featuretype::DataType)
    W = (featuretype=>Vector{ParamTriple})[]
    b = [ParamTriple() for label in labels]
    fwdlookup = [l=>i for (i,l) in enumerate(labels)]::Dict{L,Int}
    revlookup = [i=>l for (l,i) in fwdlookup]::Dict{Int,L}
    AvgPerceptronParams(W, b, fwdlookup, revlookup, 1.0)
end

function predict{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F})
    ŷ = zeros(length(θ.b))
    for i in x
        Wi = get!(()->zero(θ.b), θ.W, i)
        for j = 1:length(Wi)
            ŷ[j] += (Wi[j].total + (Wi[j].value * (θ.t - Wi[j].stamp))) / θ.t
        end
    end
    for j = 1:length(θ.b)
        ŷ[j] += (θ.b[j].total + (θ.b[j].value * (θ.t - θ.b[j].stamp))) / θ.t
    end
    indmax(ŷ)
end

function predict_current{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F})
    ŷ = [b.total for b in θ.b]
    for i in x
        Wi = get!(()->zero(θ.b), θ.W, i)
        for j = 1:length(Wi)
            ŷ[j] += Wi[j].value
        end
    end
    indmax(ŷ)
end

function update!{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F}, y::Int, ŷ::Int)
    θ.b[y].total += θ.b[y].value * (θ.t - θ.b[y].stamp)
    θ.b[ŷ].total += θ.b[ŷ].value * (θ.t - θ.b[ŷ].stamp)
    θ.b[y].stamp = θ.t
    θ.b[ŷ].stamp = θ.t
    # normal perceptron update
    θ.b[y].value += 1.0
    θ.b[ŷ].value -= 1.0
    for i in x
        Wi = get!(()->zero(θ.b), θ.W, i)
        Wi[y].total += Wi[y].value * (θ.t - Wi[y].stamp)
        Wi[ŷ].total += Wi[ŷ].value * (θ.t - Wi[ŷ].stamp)
        Wi[y].stamp = θ.t
        Wi[ŷ].stamp = θ.t
        # normal perceptron update
        Wi[y].value += 1.0
        Wi[ŷ].value -= 1.0
    end
end

function step!{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F}, ylabel::L)
    ŷcurrent = predict_current(θ, x)
    ŷavg = predict(θ, x)
    y = θ.fwdlookup[ylabel]
    if y != ŷcurrent
        update!(θ, x, y, ŷcurrent)
    end
    θ.t += 1
    (y == ŷavg) ? 0.0 : 1.0
end

end # module
