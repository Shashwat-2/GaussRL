import torch
from tqdm import tqdm
import numpy as np
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_step(batch, actor, critic, actor_optimizer, critic_optimizer, trj):
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    J = 0
    critic_loss, actor_loss = 0, 0
    for t,ep in enumerate(batch):
        ep_sum = 0
        for i,(s,a,r,s_,tcoeff,h) in enumerate(ep):

            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            s_ = s_.to(device)
            h = h.to(device) if h is not None else h

            G = sum([r*0.9**j for j,(_,_,r,_,_,_) in enumerate(ep[i:])])
            value = critic(s)[0][0]
            critic_loss += torch.nn.functional.mse_loss(G.to(device), value)

            Adv = G - value

            u,std,_ = actor(s,h)
            pi = torch.distributions.Normal(u,std)
            log_pi = pi.log_prob(a).sum()
            actor_loss += log_pi * Adv

            # ep_sum += torch.sum(-0.5*((a-u)/std)**2 - torch.log(std) - 0.5*torch.log(2*torch.tensor(torch.pi))) * \
            #           sum([r*0.9**j for j,(s,a,r,s_,_,_) in enumerate(ep[i:])]) - critic(s)
    actor_loss /= sum(len(ep) for ep in batch)
    critic_loss /= sum(len(ep) for ep in batch)
    J = actor_loss + critic_loss
    (-J).backward()
    with torch.no_grad():
        actor_optimizer.step()
        critic_optimizer.step()
        return J.detach().item(), actor_loss.detach().item(), critic_loss.detach().item()
    
def run(epochs, trj, actor, critic, actor_optimizer, critic_optimizer):
    try:
        for epoch in range(epochs):
            L = 0
            for i in tqdm(range(2020,1895,-5)):
                batch = trj.get_traj(i,20,actor)
                T, A, C = train_step(batch, actor, critic, actor_optimizer, critic_optimizer, trj)
                L += T
            print(f'Total Cost: {L/(i+1)}')
        torch.save(actor.state_dict(), f'Models/actor.pt')
        torch.save(critic.state_dict(), f'Models/critic.pt')
        print('Model saved successfully')
    except (KeyboardInterrupt):
        torch.save(actor.state_dict(), f'Models/actor.pt')
        torch.save(critic.state_dict(), f'Models/critic.pt')
        print('Model saved successfully')