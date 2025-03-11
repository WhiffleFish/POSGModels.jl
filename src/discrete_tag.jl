module DiscreteTag

using MarkovGames
using StaticArrays
using POMDPTools
using LinearAlgebra

export Coord, TagMG, TagState

const ACTION_DIRS = SA[
    SA[0, 1], # up
    SA[1, 0], # right
    SA[0,-1], # down
    SA[-1,0]  # left
]

const Coord = SVector{2, Int}

struct TagState
    pursuer::Coord
    evader::Coord
    terminal::Bool
end

Base.@kwdef struct TagMG{RM} <: MG{TagState, Tuple{Int,Int}}
    reward_model    ::  RM          = DenseReward()
    tag_reward      ::  Float64     = 10.0
    step_cost       ::  Float64     = 1.0
    discount        ::  Float64     = 0.95
    floor           ::  Coord       = Coord(11, 7)
    obstacles       ::  Set{Coord}  = Set{Coord}()
    initialstate    ::  TagState    = TagState(Coord(1,1), Coord(1,5), false)
end

Base.@kwdef struct DenseReward
    scale   ::  Float64 = 1.0
    peak    ::  Float64 = 10.0
    d_max   ::  Float64 = 5.0
end

function (dr::DenseReward)(s1::Coord, s2::Coord)
    (;scale, peak, d_max) = dr
    d = norm(s2 .- s1, 2)
    return if iszero(d)
        peak
    elseif d > d_max
        0.0
    else
        (scale/d)
    end
end

struct SparseReward end

(sr::SparseReward)(s1::Coord, s2::Coord) = all(s1 .== s2)

pursuer_actions(::TagMG) = 1:4

evader_actions(::TagMG) = 1:4

function MarkovGames.states(p::TagMG)
    v = map(CartesianIndices((p.floor[1], p.floor[2], p.floor[1], p.floor[2]))) do c_i
        t = Tuple(c_i)
        return TagState(
            Coord(t[1], t[2]), 
            Coord(t[3], t[4]), 
            false
        )
    end
    return push!(vec(v), TagState(Coord(1,1), Coord(1,1), true)) 
end

MarkovGames.actions(p::TagMG) = (pursuer_actions(p), evader_actions(p))

MarkovGames.discount(p::TagMG) = p.discount

MarkovGames.initialstate(p::TagMG) = Deterministic(p.initialstate)

function add_if_clear(floor, obstacles, s::Coord, a::Coord)
    sp = s .+ a
    if (any(sp .< Coord(1,1)) || any(sp .> floor) || sp ∈ obstacles)
        return s
    else
        return sp
    end
end

add_if_clear(floor::Coord, obstacles, s::Coord, a::Int) = add_if_clear(floor, obstacles, s, ACTION_DIRS[a])

function MarkovGames.transition(p::TagMG{DenseReward}, s, (a1, a2))
    (;floor, obstacles) = p
    next_pursuer = add_if_clear(floor, obstacles, s.pursuer, a1)
    next_evader = add_if_clear(floor, obstacles, s.evader, a2)
    return Deterministic(TagState(next_pursuer, next_evader, false))
end

function MarkovGames.transition(p::TagMG, s, (a1, a2))
    (;floor, obstacles) = p
    return if s.pursuer == s.evader # terminal -- successful tag
        Deterministic(TagState(Coord(1,1), Coord(1,1), true))
    else
        next_pursuer = add_if_clear(floor, obstacles, s.pursuer, a1)
        next_evader = add_if_clear(floor, obstacles, s.evader, a2)
        Deterministic(TagState(next_pursuer, next_evader, false))
    end
end

MarkovGames.isterminal(::TagMG, s) = s.terminal

MarkovGames.reward(p::TagMG, s::TagState, a) = isterminal(p, s) ? 0.0 : p.reward_model(s.pursuer, s.evader)

function stateindex(f::Coord, s::TagState)
    (;pursuer, evader, terminal) = s
    nc, nr = f
    n_pos = (nr^2)*(nc^2)
    if terminal
        return n_pos+1
    else
        return LinearIndices((nc,nr,nc,nr))[pursuer[1], pursuer[2], evader[1], evader[2]]
    end
end

MarkovGames.stateindex(p::TagMG, s::TagState) = stateindex(p.floor, s)

function MarkovGames.convert_s(::Type{Vector{T}} , s::TagState, p::TagMG) where T
    (; floor) = p
    pursuer = (s.pursuer .- floor ./ 2) ./ floor
    evader = (s.evader .- floor ./ 2) ./ floor
    return T[
        pursuer[1], pursuer[2],
        evader[1], evader[2]
    ]
end

function MarkovGames.convert_s(::Type{TagState}, x::AbstractVector, p::TagMG)
    (; floor) = p
    pursuer = round.(Int, (x[1:2] .* floor) .+ floor ./ 2)
    evader = round.(Int, (x[3:4] .* floor) .+ floor ./ 2)
    return TagState(Coord(pursuer[1], pursuer[2]), Coord(evader[1], evader[2]), false)
end

end
