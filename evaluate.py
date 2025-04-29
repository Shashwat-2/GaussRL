from environment import Trajectory
import torch
from policy_net import Policy
import plots

def run(year):

    trj = Trajectory()
    trj.reset()
    policy = Policy(len(trj.current_state))
    policy.load_state_dict(torch.load(f'Models/policy.pt'))
    pred_coeffs = trj.get_traj(year,1,policy)[0][-1]

    # plots.contour_plot(trj, pred_coeffs)
    # plots.gauss_coeff_plot(trj, pred_coeffs)
    # plots.obs_diff(trj, pred_coeffs)
    # plots.obs_data_loc(trj)
    # plots.obs_data_count(trj)
    return pred_coeffs