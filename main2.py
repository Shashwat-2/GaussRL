from environment import Trajectory
from policy_net import Policy
import trainer
import evaluate
import torch
    
trj = Trajectory()
trj.reset()
policy = Policy(len(trj.current_state))
optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
epochs = 1

trainer.run(epochs, trj, policy, optimizer)
# pred,true = evaluate.run(1945)[3:5]

# print(pred, true)

# LON, LAT, (F, B_r, B_theta, B_phi) = trj.contour_plot(1000,1000,true)
# div = trj.divergence(6371.2, 90-LAT, LON, B_r, B_theta, B_phi)
# print(div)

# LON, LAT, (F, B_r, B_theta, B_phi) = trj.contour_plot(1000,1000,pred)
# div = trj.divergence(6371.2, 90-LAT, LON, B_r, B_theta, B_phi)
# print(div)
