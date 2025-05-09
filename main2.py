from environment import Trajectory
from policy_net import Actor, Critic
import trainer
import evaluate
import torch
import plots
from tqdm import tqdm
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
trj = Trajectory()
trj.reset()

# actor = Actor(len(trj.current_state))
# actor.to(device)
# critic = Critic(len(trj.current_state))
# critic.to(device)

# actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
# critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

# epochs = 50

# trainer.run(epochs, trj, actor, critic, actor_optimizer, critic_optimizer)
# for year in tqdm(range(2020,1895,-5)):
pred,true = evaluate.run(1995)[3:5]
pred = pred.squeeze(0)

#     plt.plot(torch.log(abs(pred)), label='pred')
#     plt.plot(torch.log(abs(true)), label='true')
#     plt.xlabel('Coefficient id')
#     plt.ylabel('abs(Value)')
#     plt.title(f'Year {year}')
#     plt.legend()
#     plt.savefig(f'Plots/coeff_plot_{year}.png', dpi=600)
#     plt.clf()

    # plots.contour_plot(trj, pred, year)

LON, LAT, (F, B_r, B_theta, B_phi) = trj.contour_plot(1000,1000,true)
div1 = trj.divergence(6371.2, 90-LAT, LON, B_r, B_theta, B_phi)

LON, LAT, (F, B_r, B_theta, B_phi) = trj.contour_plot(1000,1000,pred)
div2 = trj.divergence(6371.2, 90-LAT, LON, B_r, B_theta, B_phi)
print(100/div1, 100/div2)

# evaluate.run(1945)
# year = 1920
# pred,true = evaluate.run(year)[3:5]
# pred = pred.squeeze(0)
# true = trj.coeff(year)

# print(trj.dipole_fraction(pred))
# print(trj.dipole_fraction(true))