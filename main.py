from environment import Trajectory
from policy_net import Policy
import trainer
import evaluate
import torch

for year in range(1900,1931,1):
    
    trj = Trajectory(year)
    trj.reset()
    policy = Policy(len(trj.current_state))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
    iterations = 1
    epochs = 10

    trainer.run(epochs, iterations, trj, policy, optimizer)
    evaluate.run(year)
