using Datasets, Perceptron, PassiveAgressive, LogReg

function mistakecounts(num::Int)
    D = Datasets.get20news()
    labels = [Set([y for (x,y) in D]...)...]
    shuffle!(D)
    θ1 = PerceptronParams(labels, ASCIIString)
    θ2 = AvgPerceptronParams(labels, ASCIIString)
    θ3 = PassiveAgressiveParams(labels, ASCIIString)
    θ4 = PassiveAgressiveParams(labels, ASCIIString)
    θ5 = LogRegParams(labels, ASCIIString)
    θ6 = LogRegParams(labels, ASCIIString)
    θ7 = LogRegParams(labels, ASCIIString)
    
    err1 = 0.0
    err2 = 0.0
    err3 = 0.0
    err4 = 0.0
    err5 = 0.0
    err6 = 0.0
    err7 = 0.0
    total = (Float64,Float64,Float64,Float64,Float64,Float64,Float64)[] 
    percent = (Float64,Float64,Float64,Float64,Float64,Float64,Float64)[]
    seen = 0
    step = 100
    for (x, y) in D
        err1 += step!(θ1, x, y)
        err2 += step!(θ2, x, y)
        err3 += step!(θ3, x, y, 0.0001)
        err4 += step!(θ4, x, y, 0.001)
        err5 += step!(θ5, x, y, 0.01)
        err6 += step!(θ6, x, y, 0.1)
        err7 += step!(θ7, x, y, 0.5)
        seen += 1
        if rem(seen, step) == 0
            println("seen: $seen")
            println("1: $err1, 2: $err2, 3: $err3, 4: $err4, 5: $err5, 6: $err6, 7: $err7")
            push!(total, (err1, err2, err3, err4, err5, err6, err7))
            push!(percent, map(c -> c / seen, (err1, err2, err3, err4, err5, err6, err7)))
        end
    end

    ftotal = open("./results/avg/total-$(lpad(num, 2, 0)).csv", "w")
    fpercent = open("./results/avg/percent-$(lpad(num, 2, 0)).csv", "w")
    for i = 1:length(total)
        write(ftotal, "$(i * step), $(join(total[i], ", "))\n")
        write(fpercent, "$(i * step), $(join(percent[i], ", "))\n")
    end
    close(ftotal)
    close(fpercent)
end
