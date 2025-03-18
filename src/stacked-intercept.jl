module StackedIntercept

using MarkovGames
using StaticArrays
using POMDPTools
using LinearAlgebra

export Coord, StackedInterceptMG, StackedInterceptState

const ACTION_DIRS = SA[
    SA[0, 1], # up
    SA[1, 0], # right
    SA[0,-1], # down
    SA[-1,0]  # left
]

const Coord = SVector{2, Int}

struct StackedInterceptState
    attackers::SVector{2,Coord} # could parameterize, but keep as 2 for now
    defender::Coord
    terminal::Bool
end

Base.@kwdef struct StackedInterceptMG{RM} <: MG{StackedInterceptState, Tuple{SVector{2,Int},Int}}
    reward_model    ::  RM              = DenseReward()
    tag_reward      ::  Float64         = 10.0
    step_cost       ::  Float64         = 1.0
    discount        ::  Float64         = 0.95
    floor           ::  Coord           = Coord(11, 7)
    obstacles       ::  Set{Coord}      = Set{Coord}()
    initialstate    ::  StackedInterceptState  = StackedInterceptState(SA[Coord(1,1),Coord(2,1)], Coord(1,5), false)
    goal            ::  Set{Coord}      = Set{Coord}([Coord(11, 7)])
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

function MarkovGames.states(p::StackedInterceptMG)
    v = map(CartesianIndices((p.floor[1], p.floor[2], p.floor[1], p.floor[2], p.floor[1], p.floor[2]))) do c_i
        t = Tuple(c_i)
        return StackedInterceptState(
            SA[Coord(t[1], t[2]), Coord(t[3], t[4])],
            Coord(t[5], t[6]), 
            false
        )
    end
    return push!(vec(v), StackedInterceptState(SA[Coord(1,1),Coord(1,1)], Coord(1,1), true)) 
end

MarkovGames.actions(p::StackedInterceptMG) = (CartesianIndices((1:4, 1:4)), CartesianIndices(1:4))

MarkovGames.discount(p::StackedInterceptMG) = p.discount

MarkovGames.initialstate(p::StackedInterceptMG) = Deterministic(p.initialstate)

function add_if_clear(floor, obstacles, s::Coord, a::Coord)
    sp = s .+ a
    if (any(sp .< Coord(1,1)) || any(sp .> floor) || sp ∈ obstacles)
        return s
    else
        return sp
    end
end

add_if_clear(floor::Coord, obstacles, s::Coord, a::Int) = add_if_clear(floor, obstacles, s, ACTION_DIRS[a])

function MarkovGames.transition(p::StackedInterceptMG{DenseReward}, s, (a1, a2))
    (;floor, obstacles) = p
    return if any(s_i ∈ p.goal for s_i in s.attackers)
        Deterministic(StackedInterceptState(SA[Coord(1,1),Coord(1,1)], Coord(1,1), true))
    else
        next_attackers = map(s.attackers, Tuple(a1)) do s_i, a_i
            add_if_clear(floor, obstacles, s_i, a_i)
        end
        next_defender = add_if_clear(floor, obstacles, s.defender, only(Tuple(a2)))
        return Deterministic(StackedInterceptState(next_attackers, next_defender, false))
    end
end

# function MarkovGames.transition(p::InterceptMG, s, (a1, a2))
#     (;floor, obstacles) = p
#     return if s.attacker == s.defender || s.attacker ∈ p.goal 
#         Deterministic(InterceptState(Coord(1,1), Coord(1,1), true))
#     else
#         next_attacker = add_if_clear(floor, obstacles, s.attacker, a1)
#         next_defender = add_if_clear(floor, obstacles, s.defender, a2)
#         Deterministic(InterceptState(next_attacker, next_defender, false))
#     end
# end

MarkovGames.isterminal(::StackedInterceptMG, s) = s.terminal

function MarkovGames.reward(p::StackedInterceptMG, s::StackedInterceptState, a)
    if isterminal(p, s)
        return 0.0
    else
        r = mapreduce(+, s.attackers) do s_i
            -p.reward_model(s_i, s.defender)
        end
        for s_i ∈ s.attackers
            s_i ∈ p.goal && (r += 10.0)
        end
        return r
    end
end

function stateindex(f::Coord, s::StackedInterceptState)
    (;attackers, defender, terminal) = s
    nc, nr = f
    if terminal
        n_agents = length(attackers) + 1
        n_pos = (nr * nc) ^ n_agents
        return n_pos+1
    else
        return LinearIndices((nc,nr,nc,nr,nc,nr))[attackers[1]..., attackers[2]..., defender...]
    end
end

MarkovGames.stateindex(p::StackedInterceptMG, s::StackedInterceptState) = stateindex(p.floor, s)

function MarkovGames.convert_s(::Type{Vector{T}} , s::StackedInterceptState, p::StackedInterceptMG) where T
    (; floor) = p
    attackers = map(s.attackers) do s_i
        (s_i .- floor ./ 2) ./ floor
    end
    defender = (s.defender .- floor ./ 2) ./ floor
    return T[
        attackers[1]..., 
        attackers[2]...,
        defender...
    ]
end

function MarkovGames.convert_s(::Type{StackedInterceptState}, x::AbstractVector, p::StackedInterceptMG)
    (; floor) = p
    # should get this from game type inference
    n_attackers = 2 # (length(x) ÷ 2) - 2
    attackers = map(StaticArrays.SUnitRange(1,n_attackers)) do idx
        idx0 = 2*(idx - 1) + 1
        idx1 = idx0 + 2
        attacker = x[idx0:idx1]
        return Coord(round.(Int, (attacker .* floor) .+ floor ./ 2))
    end
    defender = round.(Int, (x[3:4] .* floor) .+ floor ./ 2)
    return InterceptState(attackers, Coord(defender[1], defender[2]), false)
end

end
