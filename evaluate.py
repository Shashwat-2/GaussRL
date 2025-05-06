from environment import Trajectory
import torch
from policy_net import Actor
import plots
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def run(year):

    trj = Trajectory()
    trj.reset()
    policy = Actor(len(trj.current_state)).to(device)
    policy.load_state_dict(torch.load(f'Models/actor.pt'))
    pred_coeffs = trj.get_traj(year,1,policy)[0][-1]
    # plots.contour_plot(trj, pred_coeffs)
    # plots.gauss_coeff_plot(trj, pred_coeffs)
    # plots.obs_diff(trj, pred_coeffs)
    # plots.obs_data_loc(trj)
    # plots.obs_data_count(trj)
    return pred_coeffs