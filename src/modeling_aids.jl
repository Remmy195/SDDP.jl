# The function `_lattice_approximation` is derived from a function of the same name in the
# `ScenTrees.jl` package by Kipngeno Kirui and released under the MIT license.
# The reproduced function, and other functions contained only in this file, are also
# released under MIT.
#
# Copyright (c) 2019 Kipngeno Kirui <kipngenokirui1993@gmail.com>
# Copyright (c) 2019 Oscar Dowson <o.dowson@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function find_min(x::Vector{T}, y::T) where {T<:Real}
    best_i = 0
    best_z = Inf
    for i in 1:length(x)
        z = abs(x[i] - y)
        if z < best_z
            best_i = i
            best_z = z
        end
    end
    return best_z, best_i
end

function _lattice_approximation(
    f::Function,
    states::Vector{Int},
    scenarios::Int,
)
    return _lattice_approximation(
        f,
        states,
        scenarios,
        [f()::Vector{Float64} for _ in 1:scenarios],
    )
end

function _quantiles(x, N)
    if N == 1
        return [Statistics.mean(x)]
    end
    return Statistics.quantile(x, range(0.01, 0.99; length = N))
end

function _lattice_approximation(
    f::Function,
    states::Vector{Int},
    scenarios::Int,
    simulations::Vector{Vector{Float64}},
)
    simulation_matrix = reduce(hcat, simulations)
    support = map(1:length(states)) do t
        return _quantiles(@view(simulation_matrix[t, :]), states[t])
    end
    probability = [zeros(states[t-1], states[t]) for t in 2:length(states)]
    prepend!(probability, Ref(zeros(1, states[1])))
    distance = 0.0
    for (n, path) in enumerate(simulations)
        dist, last_index = 0.0, 1
        for t in 1:length(states)
            for i in 1:length(states[t])
                if sum(@view probability[t][:, i]) < 1.3 * sqrt(n) / states[t]
                    support[t][i] = path[t]
                end
            end
            min_dist, best_idx = find_min(support[t], path[t])
            dist += min_dist^2
            probability[t][last_index, best_idx] += 1.0
            support[t][best_idx] -=
                min_dist * (support[t][best_idx] - path[t]) / (3000 + n)^0.75
            last_index = best_idx
        end
        distance = (distance * (n - 1) + dist) / n
    end
    for p in probability
        p ./= sum(p; dims = 2)
        if any(isnan, p)
            p[vec(isnan.(sum(p; dims = 2))), :] .= 0.0
        end
    end
    return support, probability
end


# Lattice approximation with additional support for transitions from the last stage to the first
function extended_lattice_approximation(
    simulator::Function,
    states::Vector{Int},
    scenarios::Int,
    simulations::Vector{Vector{Float64}}
    )
    simulation_matrix = reduce(hcat, simulations)  # Create a matrix from the simulations

    # Step 1: Quantization using quantiles for each stage
    support = map(1:length(states)) do t
        compute_quantiles(collect(@view simulation_matrix[t, :]), states[t])
    end

    # Step 2: Initialize probability matrices for each stage transition
    probability = [zeros(states[t - 1], states[t]) for t in 2:length(states)]
    prepend!(probability, Ref(zeros(1, states[1])))  # Probability for root to first stage

    distance = 0.0
    # Iterate through all scenarios to assign transitions based on closest nodes
    for (n, path) in enumerate(simulations)
        dist, last_index = 0.0, 1
        for t in 1:length(states)
            min_dist, best_idx = find_minimum(support[t], path[t])
            dist += min_dist^2
            probability[t][last_index, best_idx] += 1.0
            support[t][best_idx] -= min_dist * (support[t][best_idx] - path[t]) / (3000 + n)^0.75
            last_index = best_idx
        end
        distance = (distance * (n - 1) + dist) / n
    end

    # Step 3: Add quantization and transition probabilities from last stage to initial stage
    last_stage = length(states)
    initial_stage = 1
    
    # Create a transition matrix from last stage to initial stage
    last_to_initial_prob = zeros(states[last_stage], states[initial_stage])
    
    for i in 1:states[last_stage]
        for j in 1:states[initial_stage]
            distance = norm(support[last_stage][i] - support[initial_stage][j])
            last_to_initial_prob[i, j] = 1.0 / (1.0 + distance)  # Inverse distance weighting
        end
    end
    
    # Normalize the last-to-initial transition probabilities
    for i in 1:states[last_stage]
        row_sum = sum(last_to_initial_prob[i, :])
        if row_sum > 0
            last_to_initial_prob[i, :] .= last_to_initial_prob[i, :] ./ row_sum
        end
    end

    # Step 4: Final adjustment to normalize the entire probability space
    for p in probability
        p ./= sum(p; dims = 2)
        if any(isnan, p)
            p[vec(isnan.(sum(p; dims = 2))), :] .= 0.0
        end
    end

    return support, probability, last_to_initial_prob
end


"""
    _allocate_support_budget(f, budget, scenarios)

Allocate the `budget` nodes amongst the stages for a Markovian approximation.
By default, we distribute nodes based on the relative variance of the stages.
"""
function _allocate_support_budget(
    f::Function,
    budget::Int,
    scenarios::Int,
)::Vector{Int}
    return _allocate_support_budget(
        [f()::Vector{Float64} for _ in 1:scenarios],
        budget,
        scenarios,
    )
end

function _allocate_support_budget(
    simulations::Vector{Vector{Float64}},
    budget::Int,
    scenarios::Int,
)::Vector{Int}
    stage_var = Statistics.var(simulations)
    states = ones(Int, length(stage_var))
    if budget < length(stage_var)
        @warn(
            "Budget for nodes is less than the number of stages. Using one " *
            "node per stage.",
        )
        return states
    end
    s = sum(stage_var)
    if s ≈ 0.0
        # If the sum of the variances is 0, then the simulator must be
        # deterministic. Regardless of the budget, return a single Markov state
        # for each stage.
        return states
    end
    for i in 1:length(states)
        states[i] = max(1, round(Int, budget * stage_var[i] / s))
    end
    while sum(states) != budget
        if sum(states) > budget
            states[argmax(states)] -= 1
        else
            states[argmin(states)] += 1
        end
    end
    return states
end

_allocate_support_budget(::Any, budget::Vector{Int}, ::Int) = budget






"""
    MarkovianGraph(
        simulator::Function;
        budget::Union{Int,Vector{Int}},
        scenarios::Int = 1000,
    )

Construct a Markovian graph by fitting Markov chain to scenarios generated by
`simulator()`.

`budget` is the total number of nodes in the resulting Markov chain. This can
either be specified as a single `Int`, in which case we will attempt to
intelligently distributed the nodes between stages. Alternatively, `budget` can
be a `Vector{Int}`, which details the number of Markov state to have in each
stage.
"""
function MarkovianGraph(
    simulator::Function;
    budget::Union{Int,Vector{Int}},
    scenarios::Int = 1000,
)
    scenarios = max(scenarios, 10)
    simulations = [simulator()::Vector{Float64} for _ in 1:scenarios]
    states = _allocate_support_budget(simulations, budget, scenarios)
    support, probability =
        _lattice_approximation(simulator, states, scenarios, simulations)
    g = Graph((0, 0.0))
    for (i, si) in enumerate(support[1])
        _add_node_if_missing(g, (1, si))
        _add_to_or_create_edge(g, (0, 0.0) => (1, si), probability[1][1, i])
    end
    for t in 2:length(support)
        for (j, sj) in enumerate(support[t])
            _add_node_if_missing(g, (t, sj))
            for (i, si) in enumerate(support[t-1])
                _add_to_or_create_edge(
                    g,
                    (t - 1, si) => (t, sj),
                    probability[t][i, j],
                )
            end
        end
    end
    return g
end



"""
    CyclicMarkovianGraph(
        simulator::Function;
        budget::Union{Int, Vector{Int}},
        scenarios::Int = 1000,
        cycle_prob::Float64 = 0.0,
        investment::Bool = false
    )

Construct a Cyclic Markovian graph by fitting Markov chain to scenarios generated by
`simulator()`.

`cycle_prob` is the discount factor used to normalize the nodal transitions
at the last decision nodes to the initial stage. The discount factor must be less than one.
'investment' Boolean if set to true adds a node after the root node and before the markov nodes
with no markov state. Note: the cycle transitions from the last markov node to the first markov node
not the investment.
"""


function CyclicMarkovianGraph(
    simulator::Function;
    budget::Union{Int, Vector{Int}},
    scenarios::Int = 1000,
    cycle_prob::Float64 = 0.0,
    investment::Bool = false
    )

    println("Constructing Cyclic Markovian Graph...")
    scenarios = max(scenarios, 10)

    # 1) Generate price simulations
    simulations = [simulator()::Vector{Float64} for _ in 1:scenarios]
    println("Simulations for Markov Nodes Completed...")

    # 2) Allocate states
    states = _allocate_support_budget(simulations, budget, scenarios)
    println("States Allocated...")

    # 3) Approximate support, probabilities, and transitions
    support, probability, last_to_initial_prob = extended_lattice_approximation(
        simulator, states, scenarios, simulations
    )
    println("Support and Probability Estimated...")

    # 4) Create the SDDP graph, root node
    g = Graph((0, 0.0))

    # -----------------------------------------------------------
    # Stage offsets depend on whether we have an investment node
    #
    #  - If investment = true:
    #       * Stage 1 is the "investment node"
    #       * The first Markov stage is 2
    #       * The last Markov stage in the graph is length(support) + 1
    #
    #  - If investment = false:
    #       * Root goes directly to stage 1 for the Markov chain
    #       * The last Markov stage is length(support)
    #
    # We'll define:
    #   - first_markov_stage = 2 if investment else 1
    #   - last_markov_stage  = length(support) + 1 if investment else length(support)
    # -----------------------------------------------------------
    first_markov_stage = investment ? 2 : 1
    last_markov_stage  = investment ? (length(support) + 1) : length(support)

    # -----------------------------------------------------------
    # (A) Connect the root node (stage 0) to the first Markov stage
    #     via a node with no markov state (this is the investment node)
    # -----------------------------------------------------------
    if investment
        println("Adding investment node and edge to graph...")
        # Create the investment node (stage 1) and link root => investment
        investment_node = (1, 0.0)
        _add_node_if_missing(g, investment_node)
        _add_to_or_create_edge(g, (0, 0.0) => investment_node, 1.0)

        # Connect the investment node to the first Markov states (stage = 2)
        for (i, si) in enumerate(support[1])
            node = (first_markov_stage, si)   # (2, si)
            _add_node_if_missing(g, node)
            # probability[1][1, i] is the transition from the single investment state
            _add_to_or_create_edge(g, investment_node => node, probability[1][1, i])
        end
    else
        println("Skipping investment node; connecting root node directly to first-stage nodes...")
        # Connect root => first Markov states (stage = 1)
        for (i, si) in enumerate(support[1])
            node = (first_markov_stage, si)  # (1, si)
            _add_node_if_missing(g, node)
            # probability[1][1, i] is the transition from the single root state
            _add_to_or_create_edge(g, (0, 0.0) => node, probability[1][1, i])
        end
    end

    # -----------------------------------------------------------
    # (B) Add Markov transitions for stages 2 through length(support)
    #
    # We'll loop t over "1 to length(support)-1" in the Markov sense,
    # but define how that maps to the SDDP graph stages.
    #
    # In the original code:
    #   - If investment:
    #       for t = 2 : length(support):
    #           edges from (t, si) => (t+1, sj) with probability[t][i, j]
    #   - If not investment:
    #       for t = 2 : length(support):
    #           edges from (t-1, si) => (t, sj) with probability[t][i, j]
    #
    # We'll unify them by computing 'from_stage' and 'to_stage' appropriately.
    # -----------------------------------------------------------
    println("Adding Markov nodes and edges to graph...")
    for t in 2:length(support)
        # The probability index is t (meaning transitions from stage t -> t+1 in "Markov indexing")
        # But in the SDDP graph:
        #   - If investment, 'from_stage = t, to_stage = t+1'
        #   - If no investment, 'from_stage = t-1, to_stage = t'
        from_stage = investment ? t : (t - 1)
        to_stage   = investment ? (t + 1) : t

        for (j, sj) in enumerate(support[t])
            node_to = (to_stage, sj)
            _add_node_if_missing(g, node_to)

            for (i, si) in enumerate(support[t - 1])
                node_from = (from_stage, si)
                # probability[t][i,j] is from "state i at Markov stage t-1" => "state j at Markov stage t"
                # (the "Markov stage" doesn't necessarily match the SDDP stage index exactly)
                _add_to_or_create_edge(g, node_from => node_to, probability[t][i, j])
            end
        end
    end

    # -----------------------------------------------------------
    # (C) Add cycle edges from the last Markov stage back to the first Markov stage
    # -----------------------------------------------------------
    println("Adding cycle edges to graph...")
    if cycle_prob > 0.0
        # In "Markov indexing," the last stage is length(support).
        # In the SDDP graph:
        #   - If investment => last_stage = length(support) + 1,
        #                      cycle back to stage 2
        #   - Else           => last_stage = length(support),
        #                      cycle back to stage 1
        cycle_back_stage = investment ? 2 : 1

        for (i, last_node) in enumerate(support[length(support)])
            for (j, init_node) in enumerate(support[1])
                _add_to_or_create_edge(
                    g,
                    (last_markov_stage, last_node) => (cycle_back_stage, init_node),
                    cycle_prob * last_to_initial_prob[i, j]
                )
            end
        end
    end

    println("Graph construction complete.")
    return g
end
