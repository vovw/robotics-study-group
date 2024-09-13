# Vanilla Policy gradient implementation



## short overview of VPG
- stochastic policy

π(a|s,θ)

where

Prob(action|state, params)

- change params so as to maximise rewards
- different from TD learning, no neeed for calculating Q value or V value.
- better for continuous states




## implementation


```py
# the policy network is defined as a nn
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def update_policy(log_probs, rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
```


see full code here [vpg.py](./code/vpg.py)
