module Perceptron

import Classifier: predict!, update!, step!, fit!

export PerceptronParams, AvgPerceptronParams
export predict!, update!, step!, fit!

function keymax{K}(d::Dict{K,Float64})
    state = start(d)
    kmax, vmax = next(d, state)
    k, v = kmax, vmax
    while !done(d, state)
        (k, v), state = next(d, state)
        if v > vmax
            kmax = k
            vmax = v
        end
    end
    kmax
end

# Perceptron
type PerceptronParams{F,L}
    W::Dict{F,Dict{L,Float64}}
    b::Dict{L,Float64}
    default::L
end

function PerceptronParams(featuretype::DataType, labeltype::DataType, default)
    if labeltype != typeof(default)
        error("typeof(default) and labeltype must match: got $(typeof(default)), $labeltype")
    end
    W = Dict{featuretype,Dict{labeltype,Float64}}()
    b = Dict{labeltype,Float64}()
    PerceptronParams(W, b, default)
end

function predict{F,L}(θ::PerceptronParams{F,L}, x::Vector{F})
    ŷ = (L=>Float64)[]
    for i in x
        if haskey(θ.W, i)
            for (c, wic) in θ.W[i]
                ŷ[c] = get(ŷ, c, 0) + wic
            end
        end
    end
    for c in keys(ŷ)
        ŷ[c] += θ.b[c]
    end
    (length(ŷ) > 0) ? keymax(ŷ) : θ.default
end

function update!{F,L}(θ::PerceptronParams{F,L}, x::Vector{F}, y::L, ŷ::L)
    θ.b[y] = get(θ.b, y, 0.0) + 1.0
    θ.b[ŷ] = get(θ.b, ŷ, 0.0) - 1.0
    for i in x
        Wi = get!(θ.W, i, (L=>Float64)[])
        Wi[y] = get(Wi, y, 0.0) + 1.0
        Wi[ŷ] = get(Wi, ŷ, 0.0) - 1.0
    end
end

function step!{F,L}(θ::PerceptronParams{F,L}, x::Vector{F}, y::L)
    ŷ = predict(θ, x)
    if y != ŷ
        update!(θ, x, y, ŷ)
        return 1.0
    else
        return 0.0
    end
end

function fitperceptron!{F,L}(train::Vector{(Vector{F},L)}, 
                             valid::Vector{(Vector{F},L)},
                             numit::Int)
    θ = PerceptronParams(F, L, "*NOLABEL*")
    errors_train = 0
    errors_valid = 0
    for epoch = 1:numit
        shuffle!(train)
        errors_train = 0
        for (x, y) in train
            errors_train += step!(θ, x, y)
        end
        errors_valid = 0
        for (x, y) in valid
            if y != predict(θ, x)
                errors_valid += 1
            end
        end
        println("epoch $epoch")
        println("error [train]: $(errors_train / length(train))")
        println("error [valid]: $(errors_valid / length(valid))")
    end
end

# Averaged Perceptron
type ParamTriple
    value::Float64
    total::Float64
    stamp::Float64
end
ParamTriple() = ParamTriple(0.0, 0.0, 0.0)

type AvgPerceptronParams{F,L}
    W::Dict{F,Dict{L,ParamTriple}}
    b::Dict{L,ParamTriple}
    t::Float64
    default::L
end

function AvgPerceptronParams(featuretype::DataType, labeltype::DataType, default)
    if labeltype != typeof(default)
        error("typeof(default) and labeltype must match: got $(typeof(default)), $labeltype")
    end
    W = Dict{featuretype,Dict{labeltype,ParamTriple}}()
    b = Dict{labeltype,ParamTriple}()
    AvgPerceptronParams(W, b, 1.0, default)
end

function predict{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F})
    ŷ = (L=>Float64)[]
    for i in x
        if haskey(θ.W, i)
            for (c, wic) in θ.W[i]
                ŷ[c] = get(ŷ, c, 0.0) + (wic.total + (wic.value * (θ.t - wic.stamp))) / θ.t
            end
        end
    end
    for c in keys(ŷ)
        bc = θ.b[c]
        ŷ[c] += (bc.total + (bc.value * (θ.t - bc.stamp))) / θ.t
    end
    (length(ŷ) > 0) ? keymax(ŷ) : θ.default
end

function predict_inner{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F})
    ŷ = (L=>Float64)[]
    ŷavg = (L=>Float64)[]
    for i in x
        if haskey(θ.W, i)
            for (c, wic) in θ.W[i]
                ŷ[c] = get(ŷ, c, 0.0) + wic.value
                ŷavg[c] = get(ŷavg, c, 0.0) + (wic.total / θ.t)
            end
        end
    end
    for c in keys(ŷ)
        ŷ[c] += θ.b[c].value
        ŷavg[c] += (θ.b[c].total / θ.t)
    end
    ret1 = (length(ŷ) > 0) ? keymax(ŷ) : θ.default
    ret2 = (length(ŷavg) > 0) ? keymax(ŷavg) : θ.default
    ret1, ret2
end

function update!{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F}, y::L, ŷ::L)
    by = get!(()->ParamTriple(), θ.b, y)
    bŷ = get!(()->ParamTriple(), θ.b, ŷ)
    by.total = by.value * (θ.t - by.stamp)
    bŷ.total = bŷ.value * (θ.t - bŷ.stamp)
    by.stamp = θ.t
    bŷ.stamp = θ.t
    # normal perceptron update
    by.value += 1.0
    bŷ.value -= 1.0
    for i in x
        Wi = get!(θ.W, i, (L=>ParamTriple)[])
        Wiy = get!(()->ParamTriple(), Wi, y)
        Wiŷ = get!(()->ParamTriple(), Wi, ŷ)
        Wiy.total = Wiy.value * (θ.t - Wiy.stamp)
        Wiŷ.total = Wiŷ.value * (θ.t - Wiŷ.stamp)
        Wiy.stamp = θ.t
        Wiŷ.stamp = θ.t
        # normal perceptron update
        Wiy.value += 1.0
        Wiŷ.value -= 1.0
    end
end

function step!{F,L}(θ::AvgPerceptronParams{F,L}, x::Vector{F}, y::L)
    ŷ, ŷavg = predict_inner(θ, x)
    if y != ŷ
        update!(θ, x, y, ŷ)
    end
    θ.t += 1
    return (y != ŷavg) ? 1.0 : 0.0
end

function fitavgperceptron!{F,L}(train::Vector{(Vector{F},L)},
                                valid::Vector{(Vector{F},L)},
                                numit::Int)
    θ = AvgPerceptronParams(F, L, "*NOLABEL*")
    errors_train = 0
    errors_valid = 0
    for epoch = 1:numit
        shuffle!(train)
        errors_train = 0.0
        for (x, y) in train
            errors_train += step!(θ, x, y)
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

end # module
