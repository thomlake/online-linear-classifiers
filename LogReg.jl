module LogReg

import Classifier: predict!, update!, step!

export LogRegParams
export predict!, update!, step!

function softmax!(x)
    xmax = maximum(x)
    for i = 1:length(x)
        x[i] = exp(x[i] - xmax)
    end
    Z = sum(x)
    for i = 1:length(x)
        x[i] = x[i] / Z
    end
    x
end

type LogRegParams{F,L}
    W::Dict{F,Vector{Float64}}
    b::Vector{Float64}
    fwdlookup::Dict{L,Int}
    revlookup::Dict{Int,L}
end

function LogRegParams{L}(labels::Vector{L}, featuretype::DataType)
    W = (featuretype=>Vector{Float64})[]
    b = zeros(length(labels))
    fwdlookup = [l=>i for (i,l) in enumerate(labels)]::Dict{L,Int}
    revlookup = [i=>l for (l,i) in fwdlookup]::Dict{Int,L}
    LogRegParams(W, b, fwdlookup, revlookup)
end

function prob{F,L}(θ::LogRegParams{F,L}, x::Vector{F})
    ŷ = copy(θ.b)
    for i in x
        if haskey(θ.W, i)
            ŷ += θ.W[i]
        end
    end
    softmax!(ŷ)
    ŷ
end

function predict{F,L}(θ::LogRegParams{F,L}, x::Vector{F})
    ŷ = copy(θ.b)
    for i in x
        if haskey(θ.W, i)
            ŷ += θ.W[i]
        end
    end
    θ.revlookup[indmax(ŷ)]
end

function update!{F,L}(θ::LogRegParams{F,L}, 
                      x::Vector{F}, 
                      y::Int, 
                      ŷ::Vector{Float64},
                      η::Float64)
    Δ = -η * ŷ
    Δ[y] += η
    θ.b += Δ
    for i in x
        if haskey(θ.W, i)
            θ.W[i] += Δ
        else
            θ.W[i] = copy(Δ)
        end
    end
end

function step!{F,L}(θ::LogRegParams{F,L}, x::Vector{F}, ylabel::L, η::Float64)
    ŷ = prob(θ, x)
    y = θ.fwdlookup[ylabel]
    update!(θ, x, y, ŷ, η)
    (y == indmax(ŷ)) ? 0.0 : 1.0
end

function fitlogreg!{F,L}(train::Vector{(Vector{F},L)}, 
                         valid::Vector{(Vector{F},L)},
                         numit::Int)
    trainlabels = Set([y for (x,y) in train]...)
    validlabels = Set([y for (x,y) in valid]...)
    labelset = union(trainlabels, validlabels)
    labels = [labelset...]
    θ = LogRegParams(labels, F)
    errors_train = 0.0
    errors_valid = 0.0
    for epoch = 1:numit
        shuffle!(train)
        errors_train = 0.0
        for (x, y) in train
            errors_train += step!(θ, x, y, 0.01)
        end
        errors_valid = 0.0
        for (x, y) in valid
            if y != predict(θ, x)
                errors_valid += 1.0
            end
        end
        println("epoch $epoch")
        println("error [train]: $(errors_train / length(train))")
        println("error [valid]: $(errors_valid / length(valid))")
    end
end

end
