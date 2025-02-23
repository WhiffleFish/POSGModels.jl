module DiscreteTag

using MarkovGames
using StaticArrays

export Coord, TagMG, TagState

const ACTION_DIRS = SA[
    SA[1, 0], # up
    SA[0, 1], # right
    SA[-1,0], # down
    SA[0,-1]  # left
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
    return push!(v, TagState(Coord(1,1), Coord(1,1), true)) 
end

MarkovGames.actions(p::TagMG) = (pursuer_actions(p), evader_actions(p))

MarkovGames.discount(p::TagMG) = p.discount

MarkovGames.initialstate(p::TagMG) = Deterministic(p.initialstate)

function add_if_clear(floor, obstacles, s::Coord, a::Coord)
    sp = s .+ a_dir
    if (any(sp .< Coord(1,1)) || any(sp .> floor) || sp ∈ obstacles)
        return s
    else
        return sp
    end
end

add_if_clear(floor::Coord, obstacles, s::Coord, a::Int) = add_if_clear(floor, obstacles, s, ACTION_DIRS[a])

function MarkovGames.transition(p::TagMG, s, (a1, a2))
    (;floor, obstacles) = p.pomdp
    return if s.robot == s.opponent # terminal -- successful tag
        Deterministic(TagState(Coord(1,1), Coord(1,1), true))
    else
        next_rob = add_if_clear(floor, obstacles, s.robot, a1)
        next_opp = add_if_clear(floor, obstacles, s.opponent, a2)
        Deterministic(TagState(next_rob, next_opp, false))
    end
end

MarkovGames.isterminal(::TagMG, s) = s.terminal

MarkovGames.reward(p::TagMG, s::TagState, a) = isterminal(p, s) ? 0.0 : p.reward_model(s.pursuer, s.evader)

end
