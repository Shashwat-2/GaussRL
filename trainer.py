import torch
from tqdm import tqdm
import numpy as np

def train_step(batch, policy, optimizer, trj):
    optimizer.zero_grad()
    J = 0
    for t,ep in enumerate(batch):
        ep_sum = 0
        baseline = policy.critic_forward
        for i,(s,a,r,s_,tcoeff,h) in enumerate(ep):
            u,std,_ = policy.actor_forward(s,h)
            ep_sum += torch.sum(-0.5*((a-u)/std)**2 - torch.log(std) - 0.5*torch.log(2*torch.tensor(torch.pi))) * \
                      sum([r*0.9**j for j,(s,a,r,s_,_,_) in enumerate(ep[i:])]) - baseline(s)
        J = J + (1/(t+1))*(ep_sum/(i+1) - J)
    J *= -1
    J.backward()
    with torch.no_grad():
        optimizer.step()
        return -J.item()
    
def run(epochs, trj, policy, optimizer):
    try:
        for epoch in range(epochs):
            L = 0
            for i in tqdm(range(2020,1895,-5)):
                batch = trj.get_traj(i,20,policy)
                L += train_step(batch, policy, optimizer, trj)
            print(f'Cost: {L/(i+1)}')
        torch.save(policy.state_dict(), f'Models/policy.pt')
        print('Model saved successfully')
    except (KeyboardInterrupt):
        torch.save(policy.state_dict(), f'Models/policy.pt')
        print('Model saved successfully')
