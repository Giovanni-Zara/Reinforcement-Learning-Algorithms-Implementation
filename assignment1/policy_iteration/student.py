import numpy as np

import random
import numpy as np
import gymnasium as gym


def reward_probabilities(env_size):
    rewards = np.zeros((env_size*env_size))
    i = 0
    for r in range(env_size):
        for c in range(env_size):
            state = np.array([r,c], dtype=np.uint8)
            rewards[i] = reward_function(state, env_size)
            i+=1
    return rewards

def reward_function(s, env_size):
    r = 0.0
    if (s == np.array([env_size-1, env_size-1])).all():
        r = 1
    return r

def transition_probabilities(env, s, a, env_size, directions, holes):
    cells = []
    probs = []
    prob_next_state = np.zeros((env_size, env_size))

    def check_feasibility(s_prime, s, env_size):
        if (s_prime < 0).any(): return s
                
        if s_prime[0] >= env_size: return s
        if s_prime[1] >= env_size: return s

        return s_prime

    s_prime = check_feasibility(s + directions[a, :], s, env_size)
    prob_next_state[s_prime[0], s_prime[1]] += 1/2

    s_prime = check_feasibility(s + directions[(a-1) % 4, :], s, env_size)
    prob_next_state[s_prime[0], s_prime[1]] += 1/2

    return prob_next_state

def value_iteration(env, env_size, end_state, directions, obstacles, gamma=0.99, max_iters=1000, theta=1e-3):
    # initialize
    values = np.zeros(env.observation_space.n, dtype=float)
    policy = np.zeros(env.observation_space.n, dtype=int)  # argmax actions we’ll derive
    STATES = np.zeros((env.observation_space.n, 2), dtype=np.uint8)
    REWARDS = reward_probabilities(env_size)

    # enumerate grid states in the same way
    k = 0
    for r in range(env_size):
        for c in range(env_size):
            STATES[k] = np.array([r, c], dtype=np.uint8)
            k += 1

    for i in range(max_iters):
        delta = 0.0
        v_old = values.copy()

        for s in range(env.observation_space.n):
            state = STATES[s]

            # terminal (goal) or obstacle cells have no outgoing value
            done = (state == end_state).all() or obstacles[state[0], state[1]]
            if done:
                new_v = 0.0
                best_a = policy[s] 
            else:
                best_value = -float('inf')
                best_action = 0
                for a in range(env.action_space.n):
                    next_state_prob = transition_probabilities(
                        env, state, a, env_size, directions, obstacles
                    ).flatten()
                    va = (next_state_prob * (REWARDS + gamma * v_old)).sum()
                    if va > best_value:
                        best_value = va
                        best_action = a
                new_v = best_value
                best_a = best_action

            values[s] = new_v
            policy[s] = best_a
            delta = max(delta, abs(v_old[s] - values[s]))

        if delta < theta:
            break

    print(f'finished in {i+1} iterations')
    return policy.reshape((env_size, env_size)), values.reshape((env_size, env_size))

def policy_iteration(env, env_size, end_state, directions, obstacles, gamma=0.99, max_iters=1000, theta=1e-3):
    # Initialize a random policy: for each state, pick a random action (0-3)
    policy = np.random.randint(0, env.action_space.n, (env.observation_space.n))
    
    # Initialize value function to zeros for all states
    values = np.zeros(env.observation_space.n, dtype=float)

    # Prepare state indexing: map linear state index to (row, col) coordinates
    STATES = np.zeros((env.observation_space.n, 2), dtype=np.uint8)
    # Precompute rewards for all states (1.0 at goal, 0.0 elsewhere)
    REWARDS = reward_probabilities(env_size)
    k = 0
    for r in range(env_size):
        for c in range(env_size):
            STATES[k] = np.array([r, c], dtype=np.uint8)
            k += 1

    # Main policy iteration loop: alternate between policy evaluation and improvement
    for it in range(max_iters):
        # ===== POLICY EVALUATION =====
        # Compute the value function for the current policy
        for _ in range(max_iters):
            delta = 0.0  # Track maximum change in value function
            v_old = values.copy()  # Store old values for computing updates, it stores all the old values
            
            # Update value for each state
            for s in range(env.observation_space.n):
                state = STATES[s]

                # Terminal states (goal or holes) have zero value
                done = (state == end_state).all() or obstacles[state[0], state[1]]
                if done:
                    new_v = 0.0
                else:
                    # Follow current policy: get action for this state
                    a = policy[s]
                    # Get transition probabilities for taking action a in state s
                    next_state_prob = transition_probabilities(env, state, a, env_size, directions, obstacles).flatten()
                    # Bellman expectation equation: V(s) = E[R + γ·V(s')]
                    new_v = (next_state_prob * (REWARDS + gamma * v_old)).sum() #compute new V using the old values copied before, not the newly computed ones

                values[s] = new_v
                # Track largest change for convergence check
                delta = max(delta, abs(v_old[s] - values[s]))

            # If value function converged (changes < theta), stop evaluation
            if delta < theta:
                break

        # ===== POLICY IMPROVEMENT =====
        # Update policy to be greedy with respect to current value function
        policy_stable = True  # Assume policy won't change
        
        for s in range(env.observation_space.n):
            state = STATES[s]

            # Skip terminal states (no actions to take)
            done = (state == end_state).all() or obstacles[state[0], state[1]]
            if done:
                continue

            old_action = policy[s]  # Remember current policy action
            best_value = -float('inf')
            best_action = old_action

            # Try all actions and pick the one with highest expected value
            for a in range(env.action_space.n):
                # Get transition probabilities for action a
                next_state_prob = transition_probabilities(env, state, a, env_size, directions, obstacles).flatten()
                # Compute expected value: Q(s,a) = E[R + γ·V(s')]
                va = (next_state_prob * (REWARDS + gamma * values)).sum()
                if va > best_value:
                    best_value = va
                    best_action = a

            # Update policy to best action
            policy[s] = best_action
            # If any action changed, policy is not stable
            if old_action != best_action:
                policy_stable = False

        # If policy didn't change, we've converged to optimal policy
        if policy_stable:
            break

    # Reshape from flat arrays to grid format (env_size × env_size) and return
    return policy.reshape((env_size, env_size)), values.reshape((env_size, env_size))