using POSGModels
using MarkovGames
using Random
using StaticArrays
using Test

using POSGModels.ContinuousDubin: ContinuousDubinMG, BoxActionSpace, control,
                                  action_batch, payoff_matrix, action_grid
using POSGModels.Dubin: JointDubinState, DubinMG

@testset "ContinuousDubinMG" begin
    @testset "shares its state type with the discrete game" begin
        # Reuse, not a copy: a state built for one game is the same object for the other.
        @test POSGModels.ContinuousDubin.JointDubinState === POSGModels.Dubin.JointDubinState
        s = JointDubinState(SA[1.0, 5.0, 0.0], SA[7.0, 5.0, π])
        @test s isa statetype(ContinuousDubinMG())
        @test s isa statetype(DubinMG())
    end

    @testset "action space is the unit box" begin
        game = ContinuousDubinMG()
        A1, A2 = actions(game)

        @test A1 isa BoxActionSpace{2}
        @test ndims(A1) == 2
        @test SA[0.5, 0.5] ∈ A1
        @test SA[1.2, 0.5] ∉ A1
        @test SA[-0.1, 0.5] ∉ A1
        for b in (A1, A2), _ in 1:20
            @test rand(Random.MersenneTwister(1), b) ∈ b
        end
    end

    @testset "control maps the box onto physical limits" begin
        game = ContinuousDubinMG(V=(SA[0.75, 1.5], SA[0.5, 1.0]), ω_max=(0.4, 0.2))

        @test control(game, 1, SA[0.0, 0.5]) == (0.75, 0.0)   # min speed, straight
        @test control(game, 1, SA[1.0, 0.5]) == (1.5, 0.0)    # max speed, straight
        @test control(game, 1, SA[0.5, 0.0]) == (1.125, -0.4) # mid speed, hard left
        @test control(game, 1, SA[0.5, 1.0]) == (1.125, 0.4)  # mid speed, hard right
        @test control(game, 2, SA[1.0, 1.0]) == (1.0, 0.2)    # player 2's own limits

        # Out-of-box actions clamp rather than producing illegal controls -- a flow policy
        # can land a hair outside (0,1) through float round-trips.
        @test control(game, 1, SA[1.4, -0.3]) == control(game, 1, SA[1.0, 0.0])
    end

    @testset "straight-line transition integrates at the commanded speed" begin
        game = ContinuousDubinMG(dt=1.0, V=(SA[0.75, 1.5], SA[0.5, 1.0]))
        s = JointDubinState(SA[1.0, 5.0, 0.0], SA[9.0, 1.0, π])
        # both at max speed, zero turn rate
        sp = rand(transition(game, s, (SA[1.0, 0.5], SA[1.0, 0.5])))

        @test sp.attacker ≈ SA[2.5, 5.0, 0.0]  # +1.5 in x
        @test sp.defender ≈ SA[8.0, 1.0, π]    # -1.0 in x
        @test !isterminal(game, sp)
        @test reward(game, s, (SA[1.0, 0.5], SA[1.0, 0.5]), sp) == SA[0.0, 0.0]

        # min speed moves strictly less far
        sp_slow = rand(transition(game, s, (SA[0.0, 0.5], SA[0.0, 0.5])))
        @test sp_slow.attacker[1] ≈ 1.75
        @test sp_slow.attacker[1] < sp.attacker[1]
    end

    @testset "turn rate rotates heading and curves the path" begin
        game = ContinuousDubinMG(dt=1.0, ω_max=(0.5, 0.5))
        s = JointDubinState(SA[5.0, 5.0, 0.0], SA[9.0, 9.0, 0.0])

        left  = rand(transition(game, s, (SA[0.5, 0.0], SA[0.5, 0.5]))).attacker
        right = rand(transition(game, s, (SA[0.5, 1.0], SA[0.5, 0.5]))).attacker

        @test left[3] ≈ -0.5     # θ̇ = -ω_max over dt = 1
        @test right[3] ≈ 0.5
        @test left[2] < 5.0      # turning right in θ curves -y
        @test right[2] > 5.0
        @test left[1] ≈ right[1] # symmetric in x
    end

    @testset "states are forced inside the floor" begin
        game = ContinuousDubinMG(floor=SA[10.0, 10.0])
        s = JointDubinState(SA[9.9, 5.0, 0.0], SA[0.1, 5.0, π])
        sp = rand(transition(game, s, (SA[1.0, 0.5], SA[1.0, 0.5])))

        @test sp.attacker[1] == 10.0
        @test sp.defender[1] == 0.0
        @test all(SA[0.0, 0.0] .≤ sp.attacker[SOneTo(2)] .≤ game.floor)
        @test all(SA[0.0, 0.0] .≤ sp.defender[SOneTo(2)] .≤ game.floor)
    end

    @testset "attacker reaching the goal ends the game in its favor" begin
        game = ContinuousDubinMG(tag_reward=2.5, goal=POSGModels.Dubin.CircleGoal(SA[5.0, 5.0], 1.0))
        # attacker already inside the goal
        s = JointDubinState(SA[5.0, 5.0, 0.0], SA[9.0, 9.0, π])
        a = (SA[0.5, 0.5], SA[0.5, 0.5])
        sp = rand(transition(game, s, a))

        @test isterminal(game, sp)
        @test reward(game, s, a, sp) == SA[2.5, -2.5]
    end

    @testset "capture ends the game in the defender's favor" begin
        game = ContinuousDubinMG(tag_reward=1.0, tag_radius=1.0)
        # head-on and closer than tag_radius, far from the goal
        s = JointDubinState(SA[1.0, 1.0, 0.0], SA[1.5, 1.0, π])
        a = (SA[0.5, 0.5], SA[0.5, 0.5])
        sp = rand(transition(game, s, a))

        @test isterminal(game, sp)
        @test reward(game, s, a, sp) == SA[-1.0, 1.0]
    end

    @testset "terminal states absorb" begin
        game = ContinuousDubinMG()
        s = JointDubinState(SA[1.0, 5.0, 0.0], SA[7.0, 5.0, π], true)
        sp = rand(transition(game, s, (SA[1.0, 1.0], SA[0.0, 0.0])))

        @test isterminal(game, s)
        @test sp === s
    end

    @testset "convert_s / convert_a shapes and normalization" begin
        game = ContinuousDubinMG(floor=SA[10.0, 10.0])
        s = rand(initialstate(game))
        x = MarkovGames.convert_s(Vector{Float32}, s, game)

        @test length(x) == 8
        @test x[1:2] ≈ Float32[-0.4, 0.0]                # (1,5) recentered by the floor
        @test x[3:4] ≈ Float32[0.0, 1.0]                 # sincos(0)
        @test all(-1 .≤ x[3:4] .≤ 1) && all(-1 .≤ x[7:8] .≤ 1)

        # unit-box action -> (-1,1)^2, matching convert_s's zero-centered convention
        @test MarkovGames.convert_a(Vector{Float32}, SA[0.5, 0.5], game) ≈ Float32[0.0, 0.0]
        @test MarkovGames.convert_a(Vector{Float32}, SA[0.0, 1.0], game) ≈ Float32[-1.0, 1.0]
        @test MarkovGames.convert_a(SVector{2,Float32}, SA[1.0, 0.0], game) ≈ SA[1.0f0, -1.0f0]
    end

    @testset "type stability of the hot path" begin
        game = ContinuousDubinMG()
        s = rand(initialstate(game))
        a = (SA[0.7, 0.3], SA[0.2, 0.8])

        @test @inferred(transition(game, s, a)) isa Any
        @test @inferred(reward(game, s, a, rand(transition(game, s, a)))) isa SVector{2,Float64}
        @test @inferred(control(game, 1, a[1])) isa Tuple{Float64,Float64}
    end

    @testset "payoff_matrix over sampled action batches" begin
        game = ContinuousDubinMG()
        s = rand(initialstate(game))
        rng = Random.MersenneTwister(42)
        Y1, Y2 = rand(rng, 2, 5), rand(rng, 2, 3)

        @test action_batch(Y1) == [SA[Y1[1,i], Y1[2,i]] for i in 1:5]

        P0 = payoff_matrix(game, s, Y1, Y2)                     # myopic: V ≡ 0
        P1 = payoff_matrix(game, s, Y1, Y2; V = _ -> 1.0)

        @test size(P0) == (5, 3)
        @test all(iszero, P0)                                   # no goal/capture in one step
        @test P1 ≈ fill(discount(game), 5, 3)                   # r = 0, so payoff = γ·V

        # Payoff is player 1's, and equals what the game reports pairwise.
        for i in 1:5, j in 1:3
            a = (action_batch(Y1)[i], action_batch(Y2)[j])
            sp = rand(transition(game, s, a))
            @test P0[i, j] ≈ reward(game, s, a, sp)[1]
        end

        # Terminal successors are NOT bootstrapped: at a state where the defender captures
        # on any action pair, the payoff is the raw -tag_reward regardless of V.
        s_cap = JointDubinState(SA[1.0, 1.0, 0.0], SA[1.5, 1.0, π])
        Pc = payoff_matrix(game, s_cap, Y1, Y2; V = _ -> 100.0)
        @test all(≈(-game.tag_reward), Pc)
    end

    @testset "action_grid stays in the box" begin
        G = action_grid(nv=3, nω=5)
        A1, _ = actions(ContinuousDubinMG())

        @test length(G) == 15
        @test all(a -> a ∈ A1, G)
        @test length(action_grid(nv=1, nω=1)) == 1
    end

    @testset "rollout terminates" begin
        game = ContinuousDubinMG()
        s = rand(initialstate(game))
        t = 0
        # attacker drives at max speed toward the goal; defender holds course
        while !isterminal(game, s) && t < 100
            s = rand(transition(game, s, (SA[1.0, 0.55], SA[1.0, 0.5])))
            t += 1
        end

        @test isterminal(game, s)
        @test t < 100
    end
end
