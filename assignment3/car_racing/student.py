import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    # Using discrete actions (not continuous) - the car has 5 possible actions
    continuous = False
    
    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        # 5 actions available: 0=do nothing, 1=steer left, 2=steer right, 3=gas, 4=brake
        self.n_actions = 5
        
        # === CNN FEATURE EXTRACTOR ===
        # These layers look at the 96x96 RGB image and extract useful visual features
        # First conv: takes 3 color channels, outputs 16 feature maps, shrinks image by ~4x
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        # Second conv: 16->32 feature maps, shrinks by ~2x
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Third conv: keeps 32 feature maps, small shrink
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # Flatten the 32x8x8=2048 features down to 256 for the policy/value heads
        self.fc_feature = nn.Linear(32 * 8 * 8, 256)
        
        # === ACTOR HEAD ===
        # This part decides which action to take
        self.actor_fc = nn.Linear(256, 128)  # Hidden layer for actor
        self.actor_out = nn.Linear(128, self.n_actions)  # Outputs probability for each of 5 actions
        
        # === CRITIC HEAD ===
        # This part estimates how good the current state is (used for advantage calculation)
        self.critic_fc = nn.Linear(256, 128)  # Hidden layer for critic
        self.critic_out = nn.Linear(128, 1)  # Outputs a single value estimate
        
        # === TRPO HYPERPARAMETERS ===          found a combination by trial and error and online that works "well"
        self.gamma = 0.99      # Discount factor: future rewards are worth 99% of immediate rewards
        self.lam = 0.95        # GAE lambda: controls bias-variance tradeoff in advantage estimation
        self.max_kl = 0.01     # Trust region size: don't change policy too much in one update
        self.damping = 0.1     # Numerical stability for Fisher matrix computation
        self.cg_iters = 10     # How many iterations of conjugate gradient to run (found out that 10 is enough as an approximation)
        self.critic_lr = 1e-3  # Learning rate for updating the value function
        self.critic_epochs = 5 # How many times to update critic on each batch
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize all weights using orthogonal initialization, so as random orthogonal matrices (good for deep RL)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))  #sqrt(2) gain for ReLU to keep variance     (classic initialization for ortho matrices)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Use smaller weights for actor output so initial policy is close to uniform
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.zeros_(self.actor_out.bias)

    def preprocess(self, state):
        # Convert numpy array to PyTorch tensor 
        state = torch.FloatTensor(state).to(self.device)
        # Add batch dimension if processing single image: (H,W,C) -> (1,H,W,C),     I think i'm gonna go ahead like this. NO stacking frames
        if state.dim() == 3:
            state = state.unsqueeze(0)
        # PyTorch wants (Batch, Channels, Height, Width) format, also normalize to [0,1] so / 255
        state = state.permute(0, 3, 1, 2) / 255.0
        return state

    def get_features(self, x):
        # Run image through CNN to extract visual features
        x = self.preprocess(x)
        x = F.relu(self.conv1(x))  # First conv + activation
        x = F.relu(self.conv2(x))  # Second conv + activation
        x = F.relu(self.conv3(x))  # Third conv + activation
        x = x.reshape(x.size(0), -1)  # Flatten: (B, 32, 8, 8) -> (B, 2048)
        x = F.relu(self.fc_feature(x))  # Reduce to 256 features
        return x
    
    def forward(self, x):
        # Main forward pass: get both action probabilities and state value
        features = self.get_features(x)
        
        # Actor path: features -> hidden -> action probabilities
        actor_h = F.relu(self.actor_fc(features)) #actor hidden layer + relu
        action_logits = self.actor_out(actor_h) #actor outptu, so logits
        action_probs = F.softmax(action_logits, dim=-1)  # Convert to probabilities (sum to 1)
        
        # Critic path: features -> hidden -> value estimate
        critic_h = F.relu(self.critic_fc(features))
        value = self.critic_out(critic_h)
        
        return action_probs, value
    
    def get_action_probs(self, states):
        # Get only action probabilities (skip critic computation)
        features = self.get_features(states)
        actor_h = F.relu(self.actor_fc(features))
        return F.softmax(self.actor_out(actor_h), dim=-1)
    
    def get_value(self, states):
        # Get only value estimate (skip actor computation)
        features = self.get_features(states)
        critic_h = F.relu(self.critic_fc(features))
        return self.critic_out(critic_h)
    
    def act(self, state):
        # Choose action for evaluation: pick the most probable action (greedy)
        with torch.no_grad():  # No need gradients during inference
            action_probs, _ = self.forward(state)
            action = torch.argmax(action_probs, dim=-1).item()  # Pick highest prob
        return action

    def get_actor_params(self):
        # Return all parameters that affect the policy (CNN + actor head)
        # Need CNN params because the policy depends on visual feature extraction
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + \
               list(self.conv3.parameters()) + list(self.fc_feature.parameters()) + \
               list(self.actor_fc.parameters()) + list(self.actor_out.parameters())
    
    def get_critic_params(self):
        # Return only critic-specific parameters (not CNN, actor updates that)
        return list(self.critic_fc.parameters()) + list(self.critic_out.parameters())

    def compute_gae(self, rewards, values, dones):
        # Compute Generalized Advantage Estimation
        # Advantage tells us how much better an action was compared to average
        advantages = []
        gae = 0
        values = values + [0]  # Append 0 for terminal state value
        
        # Work backwards through the trajectory
        for t in reversed(range(len(rewards))):
            # TD error: actual reward + discounted next value - predicted current value
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            # GAE accumulates TD errors with exponential decay
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae          #basically accumulated discounted TD
            advantages.insert(0, gae)
        
        # Returns are what the critic should predict (used as training targets)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]      # A = R - V => R = A + V
        return advantages, returns

    def flat_grad(self, grads):
        # Flatten all gradient tensors into one big 1D vector
        return torch.cat([g.reshape(-1) for g in grads if g is not None])

    def get_flat_params(self, params):
        # Flatten all parameter tensors into one big 1D vector
        return torch.cat([p.data.reshape(-1) for p in params])

    def set_flat_params(self, params, flat_params):
        # Set parameters from a flat 1D vector (inverse of get_flat_params)
        idx = 0
        for p in params:
            size = p.numel()  # Number of elements in this parameter tensor
            p.data.copy_(flat_params[idx:idx+size].reshape(p.shape))
            idx += size

    def compute_kl(self, old_probs, new_probs):
        # KL divergence measures how different two probability distributions are
        # KL(old || new) = sum(old * log(old/new))
        return (old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10))).sum(dim=1).mean()

    def fisher_vector_product(self, states, vector, old_probs):
        # Compute Fisher matrix times a vector without explicitly forming the matrix (fisher matrix -> 2nd derivative of KL). Compute Fxv for any vector without actually computing Fisher
        # Fisher matrix captures the curvature of the policy - how sensitive it is to parameter changes
        
        # Get current policy probabilities (need gradients)
        new_probs = self.get_action_probs(states)
        
        # Compute KL divergence between old and new policy
        kl = self.compute_kl(old_probs, new_probs)
        
        # First derivative: gradient of KL with respect to policy parameters
        actor_params = self.get_actor_params()
        kl_grads = torch.autograd.grad(kl, actor_params, create_graph=True)
        flat_kl_grad = self.flat_grad(kl_grads)
        
        # Compute dot product of gradient with the input vector
        kl_v = (flat_kl_grad * vector).sum()
        
        # Second derivative: this gives us Fisher matrix times vector
        hvp = torch.autograd.grad(kl_v, actor_params, retain_graph=True)
        flat_hvp = self.flat_grad(hvp)
        
        # Add damping for numerical stability (I need positive definite F to solve fx = b with conj grad)
        return flat_hvp + self.damping * vector

    def conjugate_gradient(self, states, b, old_probs):
        # Solve the linear system F @ x = b (F is the Fisher matrix)
        # This finds the natural gradient direction efficiently without computing F explicitly (impossible, too huge)
        x = torch.zeros_like(b)  # Start with zero solution
        r = b.clone()  # Residual: how far from solution
        p = b.clone()  # Search direction
        rdotr = torch.dot(r, r)  # Squared norm of residual
        
        for _ in range(self.cg_iters):
            # Compute F @ p using Fisher-vector product
            Ap = self.fisher_vector_product(states, p, old_probs)
            # Compute optimal step size along search direction
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            # Update solution
            x = x + alpha * p
            # Update residual
            r = r - alpha * Ap
            # Check if converged
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            # Update search direction using conjugate gradient formula
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def surrogate_loss(self, states, actions, advantages, old_probs):
        # Surrogate objective that TRPO maximizes
        # It's the expected advantage weighted by probability ratio
        new_probs = self.get_action_probs(states)
        # Get probability of the actions that were actually taken
        old_action_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze()
        new_action_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Importance sampling ratio: new_prob / old_prob
        ratio = new_action_probs / (old_action_probs + 1e-10)
        # Surrogate = average of (ratio * advantage)
        return (ratio * advantages).mean()

    def update_policy(self, states, actions, advantages):
        # Main TRPO policy update
        # Convert lists to tensors, gave error before, I redo all
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages to have zero mean and unit variance (stabilizes training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Save old policy probabilities (frozen, no gradients)
        with torch.no_grad():
            old_probs = self.get_action_probs(states).clone()
        
        actor_params = self.get_actor_params()
        
        # Clear any existing gradients
        for p in actor_params:
            if p.grad is not None:
                p.grad.zero_()
        
        # Compute surrogate loss and its gradient
        surr_loss = self.surrogate_loss(states, actions, advantages, old_probs)
        grads = torch.autograd.grad(surr_loss, actor_params, retain_graph=True)
        policy_gradient = self.flat_grad(grads)
        
        # Skip update if gradient is essentially zero
        if torch.norm(policy_gradient) < 1e-8:
            return
        
        # Use conjugate gradient to find natural gradient: F^{-1} @ g
        # This accounts for the curvature of the policy space
        step_dir = self.conjugate_gradient(states, policy_gradient, old_probs)
        
        # Compute maximum step size that satisfies KL constraint
        # Formula: sqrt(2 * max_kl / (step_dir^T @ F @ step_dir))
        sAs = torch.dot(step_dir, self.fisher_vector_product(states, step_dir, old_probs)) #sAs step^T A step
        if sAs <= 0:
            return  # Curvature is non-positive, skip update
        
        max_step = torch.sqrt(2 * self.max_kl / (sAs + 1e-8))       #solving the constraint for step size
        full_step = max_step * step_dir
        
        # Line search: try full step, then half, then quarter, etc.
        old_params = self.get_flat_params(actor_params)
        
        with torch.no_grad():
            old_surr = self.surrogate_loss(states, actions, advantages, old_probs).item()
        
        # Try progressively smaller steps until find one that works
        for i in range(10):
            alpha = 0.5 ** i  # 1, 0.5, 0.25, 0.125, ...
            new_params = old_params + alpha * full_step
            self.set_flat_params(actor_params, new_params)
            
            with torch.no_grad():
                new_probs = self.get_action_probs(states)
                new_surr = self.surrogate_loss(states, actions, advantages, old_probs).item()
                kl = self.compute_kl(old_probs, new_probs).item()
            
            # Accept step if: 1) KL constraint satisfied, 2) surrogate improved
            if kl < self.max_kl and new_surr > old_surr:
                return
        
        # If no valid step found, revert to old parameters (no update this time)
        self.set_flat_params(actor_params, old_params)

    def update_critic(self, states, returns):
        # Update value function using simple gradient descent on MSE loss
        states = torch.FloatTensor(np.array(states)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Create optimizer for critic parameters only
        optimizer = torch.optim.Adam(self.get_critic_params(), lr=self.critic_lr)
        
        # Multiple gradient steps on the same batch
        for _ in range(self.critic_epochs):
            values = self.get_value(states).squeeze()
            loss = F.mse_loss(values, returns)  # Minimize squared error
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.get_critic_params(), 0.5)  # Prevent gradient explosion
            optimizer.step()

    def train(self):
        # Main training loop
        env = gym.make('CarRacing-v2', continuous=False)
        
        n_episodes = 500
        batch_size = 2048  # Update policy after collecting this many steps
        max_steps = 1000   # Maximum steps per episode
        
        # Storage for trajectory data
        all_states, all_actions, all_rewards, all_dones, all_values = [], [], [], [], []
        
        best_reward = -float('inf')
        reward_history = []
        
        print("Starting TRPO training...")
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            negative_reward_counter = 0
            
            # Skip first 50 frames (camera zoom animation)
            for _ in range(50):
                state, _, _, _, _ = env.step(0)
            
            for step in range(max_steps):
                with torch.no_grad():
                    action_probs, value = self.forward(state)
                    
                    # Sample action from probability distribution (stochastic for exploration)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().item()
                
                # Execute action in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Track consecutive negative rewards (car might be stuck, it happened to do better with this)
                if reward < 0:
                    negative_reward_counter += 1
                else:
                    negative_reward_counter = 0
                
                # If stuck for too long, end episode with penalty, cuz the car was basically still. fix
                if negative_reward_counter > 100:
                    reward -= 10
                    terminated = True
                
                done = terminated or truncated
                
                # Store this transition
                all_states.append(state)
                all_actions.append(action)
                all_rewards.append(reward)
                all_dones.append(float(done))
                all_values.append(value.item())
                
                episode_reward += reward
                state = next_state
                
                # Update policy when batch is full
                if len(all_states) >= batch_size:
                    # Compute advantages and returns
                    advantages, returns = self.compute_gae(all_rewards, all_values, all_dones)
                    # Update actor using TRPO
                    self.update_policy(all_states, all_actions, advantages)
                    # Update critic using gradient descent
                    self.update_critic(all_states, returns)
                    # Clear storage for next batch
                    all_states, all_actions, all_rewards, all_dones, all_values = [], [], [], [], []
                
                if done:
                    break
            
            # Track rolling average of rewards
            reward_history.append(episode_reward)
            if len(reward_history) > 50:
                reward_history = reward_history[-50:]
            mean_reward = np.mean(reward_history)
            
            # Print progress every 5 episodes
            if episode % 5 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Mean(50): {mean_reward:.2f}")
            
            # Save model if it's the best so far
            if episode >= 50 and mean_reward > best_reward:
                best_reward = mean_reward
                self.save()
                print(f"New best: {best_reward:.2f}")
            
            # Early stopping if target reached
            if mean_reward > 900:
                print("Target achieved!")
                self.save()
                break
        
        env.close()
        print(f"Training complete. Best: {best_reward:.2f}")

    def save(self):
        # Save all model weights to file
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        # Load model weights from file
        self.load_state_dict(torch.load('model.pt', map_location=self.device, weights_only=True))

    def to(self, device):
        # Move model to specified device (CPU or GPU) and remember which device
        ret = super().to(device)
        ret.device = device
        return ret
