module SimpleGame

using MarkovGames
using Random
using POMDPTools

export SimpleMG

struct SimpleMG <: MG{Int, Tuple{Int, Int}}
    rewards::Vector{Matrix{Float64}}
    discount::Float64
    function SimpleMG(na1, na2; rng=Random.default_rng(), discount=0.95)
        n_mat = na1 * na2 + 1
        new([randn(rng, na1, na2) for i ∈ 1:n_mat], discount)
    end
    function SimpleMG(v::Vector{Matrix{Float64}}; discount=0.95)
        new(v, discount)
    end
end

MarkovGames.initialstate(::SimpleMG) = Deterministic(1)

MarkovGames.reward(p::SimpleMG, s, a::Tuple) = p.rewards[s][a...]

MarkovGames.actions(p::SimpleMG) = axes(first(p.rewards))

MarkovGames.states(p::SimpleMG) = 0:lastindex(p.rewards)

MarkovGames.discount(p::SimpleMG) = p.discount

function MarkovGames.transition(p::SimpleMG, s, a)
    return if isone(s)
        LinearIndices(first(p.rewards))[CartesianIndex(a)] + 1
    else
        0
    end |> Deterministic
end

MarkovGames.isterminal(::SimpleMG, s) = iszero(s)

MarkovGames.stateindex(::SimpleMG, s) = s + 1

function MarkovGames.convert_s(::Type{Vector{T}}, s::Int, p::SimpleMG) where T
    v = zeros(T, length(states(p)))
    v[s + 1] = one(T)
    return v
end

function MarkovGames.convert_s(::Type{Int}, v::AbstractVector, p::SimpleMG)
    return findfirst(isone, v) - 1
end

end
