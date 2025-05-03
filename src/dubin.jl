module Dubin

using MarkovGames
using StaticArrays
using POMDPTools
using LinearAlgebra
using RecipesBase

export DubinState, JointDubinState, DubinMG

const Vec2 = SVector{2, Float64}
const Vec3 = SVector{3, Float64}
const DubinState = Vec3

struct JointDubinState
    attacker::DubinState
    defender::DubinState
    terminal::Bool
    JointDubinState(attacker, defender, terminal=false) = new(attacker, defender, terminal)
end

struct CircleGoal
    center::Vec2
    radius::Float64
end

position(x::DubinState) = x[SOneTo(2)]
Base.in(x::Vec2, g::CircleGoal) = norm(x .- g.center, 2) < g.radius
Base.in(x::Vec3, g::CircleGoal) = position(x) in g


Base.@kwdef struct DubinMG{RM, G} <: MG{JointDubinState, Tuple{Int,Int}}
    reward_model    ::  RM                          = DenseReward()
    actions         ::  NTuple{2, Vector{Float64}}  = (deg2rad.([-30, 0, 30]), deg2rad.([-30, 0, 30]))
    V               ::  NTuple{2, Float64}          = (1.5, 1.0)
    tag_reward      ::  Float64                     = 10.0
    discount        ::  Float64                     = 0.95
    floor           ::  Vec2                        = SA[11.0, 7.0]
    initialstate    ::  JointDubinState             = JointDubinState(Vec3(1,3,0), Vec3(4,3,π/2))
    goal            ::  G                           = CircleGoal(SA[5.0,5.0], 1.0)
end

function rk4step(f, x, h)
    k1 = f(x)
    k2 = f(@.(x + h * k1 / 2))
    k3 = f(@.(x + h * k2 / 2))
    k4 = f(@.(x + h * k3))
    return @. x + h / 6 * (k1 + 2 * k2 + 2k3 + k4)
end

function dubin_dx(x, V, θ̇)
    θ = x[3]
    return SA[
        V * cos(θ),
        V * sin(θ),
        θ̇
    ]
end

function dubinstep(x::DubinState, a, V, dt)
    dx = _x -> dubin_dx(_x, V, a)
    return rk4step(dx, x, dt)
end

dubinstep(game::DubinMG, x::DubinState, a, V) = dubinstep(x, a, V, game.dt)

MarkovGames.actions(p::DubinMG) = eachindex.(p.actions)

MarkovGames.discount(p::DubinMG) = p.discount

MarkovGames.initialstate(p::DubinMG) = Deterministic(p.initialstate)

function force_inbounds(x::DubinState, floor)
    return SA[
        clamp(x[1], 0, floor[1]),
        clamp(x[2], 0, floor[2]),
        x[3]
    ]
end

function MarkovGames.transition(p::DubinMG, s::JointDubinState, (a1, a2))
    return if s.attacker ∈ p.goal || norm(s.attacker .- s.defender) < 1.0
        Deterministic(JointDubinState(s.attacker, s.defender, true))
    else
        next_attacker = force_inbounds(dubinstep(p, s.attacker, a1, game.V[1]), p.floor)
        next_defender = force_inbounds(dubinstep(p, s.defender, a2, game.V[2]), p.floor)
        Deterministic(JointDubinState(next_attacker, next_defender, false))
    end
end

MarkovGames.isterminal(::DubinMG, s) = s.terminal

function MarkovGames.reward(p::DubinMG, s::JointDubinState, a)
    if isterminal(p, s)
        return 0.0
    else
        if s.attacker ∈ p.goal
            return 10.0
        elseif norm(s.attacker .- s.defender) < 1.0
            return -10.0
        else
            return 0.0
        end
    end
end

function MarkovGames.convert_s(::Type{Vector{T}} , s::JointDubinState, p::DubinMG) where T
    (; floor) = p
    attacker_pos = (position(s.attacker) .- floor ./ 2) ./ floor
    defender_pos = (position(s.defender) .- floor ./ 2) ./ floor
    return T[
        attacker_pos..., attacker[3],
        defender_pos..., defender[3]
    ]
end

## visualization

function action_lines(game::DubinMG, x::DubinState, player::Int; n=5)
    V = game.V[player]
    A = game.actions[player]
    dts = range(0, game.dt, length=n)
    return map(A) do a
        pts = map(dts) do dt
            sp = dubinstep(x, a, V, dt)
            sp[1], sp[2]
        end
        first.(pts), last.(pts)
    end
end

@recipe function f(game::DubinMG, s::DubinState)
    (;attacker, defender) = s
    xlims --> (0, game.floor[1]+1)
    ylims --> (0, game.floor[2]+1)
    xticks --> nothing
    yticks --> nothing
    @series begin
        game.goal
    end
    @series begin
        seriestype  := :scatter
        c           --> [:blue,:red]
        [attacker[1], defender[1]], [attacker[2], defender[2]]
    end
end

@recipe function f(game::DubinMG, s::DubinState, σ1::AbstractVector, σ2::AbstractVector)
    (;attacker, defender) = s
    pol1 = σ1 |> permutedims
    pol2 = σ2 |> permutedims
    @series begin
        c       --> 1
        lw      --> 10
        alpha   --> pol1
        action_lines(game, attacker, 1)
    end
    @series begin
        c       --> :red
        lw      --> 10
        alpha   --> pol2
        action_lines(game, defender, 2)
    end
    @series begin
        game, s
    end
end

function circleshape(h,k,r)
    t = range(0, 2π, length=100)
    x = h .+ r * cos.(t)
    y = k .+ r * sin.(t)
    return x, y
end

@recipe function f(g::CircleGoal)
    seriestype := :shape
    c --> :yellow
    circleshape(g.center[1], g.center[2], g.radius)
end

@recipe f(game::DubinMG, s::DubinState, σ1::SparseCat, σ2::SparseCat) = game, s, σ1.probs, σ2.probs

@recipe f(game::DubinMG, s::DubinState, σ::ProductDistribution) = game, s, σ[1], σ[2]

end
