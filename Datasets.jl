module Datasets

export approxsplit
export dumpsplits, readsplits

function makebow(doc)
    words = split(doc, " ")
    s = Set{eltype(words)}()
    for word in words
        push!(s, word)
    end
    [s...]
end

function get20news()
    # return an array of (x,y) pairs where 
    # x is a document and
    # y is the topic
    lines = split(readall(open("./data/20news.txt")), "\n")
    
    lbls = lines[1:2:end-1]
    #docs = map(split, lines[2:2:end])
    docs = map(makebow, lines[2:2:end])

    @assert length(lbls) == length(docs)

    convert(Vector{(Vector{ASCIIString},ASCIIString)},collect(zip(docs, lbls)))
end

function approxsplit(things, p)
    # split things into length(p) chunks such that 
    # length(chunks[i]) ≈ length(things) * p[i]
    chunks = (typeof(things))[]
    l = length(things)
    s1 = 0
    for i = 1:length(p) - 1
        s2 = ifloor(s1 + l * p[i])
        push!(chunks, things[s1 + 1:s2])
        s1 = s2
    end
    push!(chunks, things[s1:end])

    chunks
end

end
