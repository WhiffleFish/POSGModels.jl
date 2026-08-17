module ContinuousDubin

using MarkovGames
using StaticArrays
using POMDPTools
using LinearAlgebra
using RecipesBase
using Random

# Everything kinematic and geometric is shared with the discrete game: the state type, the
# goal, the RK4 Dubin's step, the swept-segment capture check. Reusing the *bindings* (rather
# than copying the code) means `Dubin.JointDubinState === ContinuousDubin.JointDubinState`, so
# a state, a `CircleGoal`, or a trajectory moves between the two games untouched -- and the
# `CircleGoal` plot recipe, being registered on the type, comes along for free.
using ..Dubin: DubinState, JointDubinState, CircleGoal, Vec2, Vec3,
               position, dubinstep, force_inbounds, closest_distance, lerp

export DubinState, JointDubinState, ContinuousDubinMG, BoxActionSpace, control

## continuous action space

"""
    BoxActionSpace{N}(lo, hi)

Product of `N` closed intervals. Actions of `ContinuousDubinMG` live in the *unit* box
`(0,1)^N`, which is exactly the support of the normalizing-flow policies in
DifferentialMGs (`BoxFlowPolicy` / `BoxSplinePolicy`), so a flow's samples are valid
actions with no rescaling at the solver boundary. The map from the unit box to physical
controls `(v, θ̇)` lives in the game ([`control`](@ref)), not in the policy.
"""
struct BoxActionSpace{N}
    lo::SVector{N, Float64}
    hi::SVector{N, Float64}
end

BoxActionSpace(lo::AbstractVector, hi::AbstractVector) =
    BoxActionSpace(SVector{length(lo), Float64}(lo), SVector{length(hi), Float64}(hi))

unitbox(::Val{N}) where N = BoxActionSpace(zeros(SVector{N, Float64}), ones(SVector{N, Float64}))
unitbox(N::Int) = unitbox(Val(N))

Base.eltype(::Type{BoxActionSpace{N}}) where N = SVector{N, Float64}
Base.eltype(b::BoxActionSpace) = eltype(typeof(b))
Base.ndims(::Type{BoxActionSpace{N}}) where N = N
Base.ndims(b::BoxActionSpace) = ndims(typeof(b))
Base.length(b::BoxActionSpace) = ndims(b) # dimension, NOT cardinality -- the space is uncountable
bounds(b::BoxActionSpace) = (b.lo, b.hi)
Base.extrema(b::BoxActionSpace) = bounds(b)
Base.in(a::AbstractVector, b::BoxActionSpace) = all(b.lo .≤ a .≤ b.hi)
Base.rand(rng::AbstractRNG, b::BoxActionSpace{N}) where N =
    b.lo .+ (b.hi .- b.lo) .* rand(rng, SVector{N, Float64})
Base.rand(b::BoxActionSpace) = rand(Random.default_rng(), b)
Base.clamp(a::AbstractVector, b::BoxActionSpace) = clamp.(a, b.lo, b.hi)

const DubinAction = Vec2 # (velocity, turn rate), normalized to (0,1)^2

"""
    ContinuousDubinMG

Continuous-action variant of `Dubin.DubinMG`: each player controls both its speed and its
turn rate. An action is a point of the unit box `(0,1)^2`, mapped affinely onto
`[V[p][1], V[p][2]] × [-ω_max[p], ω_max[p]]` by [`control`](@ref). State, dynamics, capture
geometry and reward semantics (sparse ±1 on attacker-reaches-goal / capture) are those of
`Dubin.DubinMG`; only the action space differs.
"""
Base.@kwdef struct ContinuousDubinMG{G} <: MG{JointDubinState, Tuple{DubinAction, DubinAction}}
    V               ::  NTuple{2, Vec2}            = (SA[0.75, 1.5], SA[0.5, 1.0])   # (min, max) speed
    ω_max           ::  NTuple{2, Float64}         = (deg2rad(45), deg2rad(45))
    tag_reward      ::  Float64                    = 1.0
    tag_radius      ::  Float64                    = 1.0
    discount        ::  Float64                    = 0.95
    floor           ::  Vec2                       = SA[10.0, 10.0]
    initialstate    ::  JointDubinState            = JointDubinState(Vec3(1,5,0), Vec3(7,5,π))
    goal            ::  G                          = CircleGoal(SA[5.0,5.0], 1.0)
    dt              ::  Float64                    = 1.0
end

MarkovGames.actions(::ContinuousDubinMG) = (unitbox(Val(2)), unitbox(Val(2)))

MarkovGames.discount(p::ContinuousDubinMG) = p.discount

MarkovGames.initialstate(p::ContinuousDubinMG) = Deterministic(p.initialstate)

MarkovGames.isterminal(::ContinuousDubinMG, s) = s.terminal

"""
    control(game, player, a) -> (v, θ̇)

Physical controls for normalized action `a ∈ (0,1)^2`. `a` is clamped, so a policy that
puts mass exactly on the boundary (or a hair outside it, from float round-trips through a
flow) is still well defined.
"""
function control(game::ContinuousDubinMG, player::Int, a::AbstractVector)
    vmin, vmax = game.V[player]
    ω = game.ω_max[player]
    av = clamp(a[1], 0.0, 1.0)
    aω = clamp(a[2], 0.0, 1.0)
    return lerp(vmin, vmax, av), lerp(-ω, ω, aω)
end

# `Dubin.dubinstep(x, θ̇, V, dt)` is the shared RK4 integration; the only new thing here is
# that V comes out of the action rather than out of the game.
function dubin_step(game::ContinuousDubinMG, x::DubinState, player::Int, a, dt=game.dt)
    V, θ̇ = control(game, player, a)
    return dubinstep(x, θ̇, V, dt)
end

function MarkovGames.transition(game::ContinuousDubinMG, s::JointDubinState, (a1, a2))
    isterminal(game, s) && return Deterministic(s)
    next_attacker = force_inbounds(dubin_step(game, s.attacker, 1, a1), game.floor)
    next_defender = force_inbounds(dubin_step(game, s.defender, 2, a2), game.floor)
    _sp = JointDubinState(next_attacker, next_defender, false)
    terminal = s.attacker ∈ game.goal || closest_distance(s, _sp) < game.tag_radius
    return Deterministic(JointDubinState(next_attacker, next_defender, terminal))
end

function MarkovGames.reward(p::ContinuousDubinMG, s::JointDubinState, a, sp::JointDubinState)
    r = if s.attacker ∈ p.goal
        p.tag_reward
    elseif closest_distance(s, sp) < p.tag_radius
        -p.tag_reward
    else
        0.0
    end
    return SA[Float64(r), -Float64(r)]
end

function MarkovGames.convert_s(::Type{Vector{T}}, s::JointDubinState, p::ContinuousDubinMG) where T
    (; floor) = p
    attacker_pos = (position(s.attacker) .- floor ./ 2) ./ floor
    defender_pos = (position(s.defender) .- floor ./ 2) ./ floor
    return T[
        attacker_pos..., sincos(s.attacker[3])...,
        defender_pos..., sincos(s.defender[3])...,
    ]
end

# Actions are already normalized to (0,1)^2; recentered to (-1,1)^2 for network input, to
# match the zero-centered convention of `convert_s` above.
MarkovGames.convert_a(::Type{Vector{T}}, a::AbstractVector, ::ContinuousDubinMG) where T =
    T[(2 .* a .- 1)...]

MarkovGames.convert_a(::Type{SVector{2,T}}, a::AbstractVector, ::ContinuousDubinMG) where T =
    SVector{2,T}(2 .* a .- 1)

## payoff over sampled action batches
#
# `DifferentialMGs.NFGContinuousMMD.solve` wants `payoff(Y1, Y2) -> N×M`, with `Yᵖ` of size
# `d×Nᵖ` in the unit box: player 1's payoff for every *pair* of sampled actions. At a single
# state that is the one-step reward plus a discounted continuation value; the value function
# is supplied by the caller (a critic, or `s -> 0.0` for the myopic game).

"""
    action_batch(Y) -> Vector{Vec2}

Columns of a `2×N` batch of unit-box actions as static vectors.
"""
action_batch(Y::AbstractMatrix) = [Vec2(Y[1, i], Y[2, i]) for i in axes(Y, 2)]

"""
    payoff_matrix(game, s, Y1, Y2; V = _ -> 0.0)

Player 1's `N×M` payoff matrix at `s` for the sampled unit-box actions in the columns of
`Y1` (`2×N`) and `Y2` (`2×M`): `r₁(s,a,s') + γ·V(s')`, with `V` evaluated only at
non-terminal successors. Pass directly to a continuous-MMD solver as
`(Y1, Y2) -> payoff_matrix(game, s, Y1, Y2; V)`.
"""
function payoff_matrix(game::ContinuousDubinMG, s::JointDubinState,
                       Y1::AbstractMatrix, Y2::AbstractMatrix; V = _ -> 0.0)
    A1, A2 = action_batch(Y1), action_batch(Y2)
    γ = discount(game)
    return [
        begin
            a = (a1, a2)
            sp = rand(transition(game, s, a)) # deterministic dynamics
            r = reward(game, s, a, sp)[1]
            isterminal(game, sp) ? r : r + γ * V(sp)
        end
        for a1 in A1, a2 in A2
    ]
end

## visualization
#
# The `CircleGoal` recipe is inherited from `Dubin` -- recipes register on the argument type,
# and the type is the same one.

"""
Trajectory (xs, ys) traced over one `dt` for each unit-box action in `A`.
"""
function action_lines(game::ContinuousDubinMG, x::DubinState, player::Int,
                      A::AbstractVector; n=5)
    dts = range(0, game.dt, length=n)
    return map(A) do a
        pts = map(dts) do dt
            sp = dubin_step(game, x, player, a, dt)
            sp[1], sp[2]
        end
        first.(pts), last.(pts)
    end
end

action_lines(game::ContinuousDubinMG, x::DubinState, player::Int, Y::AbstractMatrix; kwargs...) =
    action_lines(game, x, player, action_batch(Y); kwargs...)

"""
Grid of unit-box actions, for visualizing the action space without a policy sample.
"""
function action_grid(; nv=3, nω=5)
    vs = nv == 1 ? [0.5] : range(0, 1, length=nv)
    ωs = nω == 1 ? [0.5] : range(0, 1, length=nω)
    return [Vec2(v, ω) for v in vs for ω in ωs]
end

@recipe function f(game::ContinuousDubinMG, s::JointDubinState)
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
        labels  --> nothing
        [attacker[1], defender[1]], [attacker[2], defender[2]]
    end
end

"""
Sampled-policy view: `A1`/`A2` are batches of unit-box actions (a `2×N` matrix, e.g. the `Y`
field of a flow policy's `rollout`, or a vector of actions). Each sampled action is drawn as
one translucent arc, so the density of arcs *is* the plotted density -- the continuous
analogue of the per-action alpha used for the discrete game.
"""
@recipe function f(game::ContinuousDubinMG, s::JointDubinState, A1, A2; alpha=0.15, weights=nothing)
    (;attacker, defender) = s
    w1, w2 = weights === nothing ? (alpha, alpha) : (permutedims(weights[1]), permutedims(weights[2]))
    @series begin
        c       --> 1
        lw      --> 4
        alpha   --> w1
        labels  --> nothing
        action_lines(game, attacker, 1, A1)
    end
    @series begin
        c       --> :red
        lw      --> 4
        alpha   --> w2
        labels  --> nothing
        action_lines(game, defender, 2, A2)
    end
    @series begin
        game, s
    end
end

end
