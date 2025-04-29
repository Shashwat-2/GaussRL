import pandas as pd
import numpy as np
from typing import List
import torch
from chaosmagpy.model_utils import synth_values
from tqdm import tqdm
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MDP:

    @torch.no_grad()
    def __init__(self) -> None:
        self.coeff = lambda year: torch.from_numpy(pd.read_excel('igrf13coeffs.xls', header=3)[5*(year//5)].to_numpy()).float()
        self.current_state = self.coeff(2020)
        self.obs = pd.read_table('results (1).tsv')[['year','lat','lon','inten','inc','decl']].dropna().to_numpy()
        self.mean_trend = np.mean(np.array([pd.read_excel('igrf13coeffs.xls', header=3)[y].to_numpy() for y in range(1900,2020,5)]), axis=0)
        self.decay = abs(self.mean_trend)/10
        self.catalog = lambda year: {(i[1], i[2]):(i[3], self.field(i[1], i[2])[0]) for i in self.obs if i[0]==year}
        self.var_catalog = lambda year, coeff: {(i[1], i[2]):(i[3], self.field(i[1], i[2], coeff(year))[0]) for i in self.obs if i[0]==year}
        self.reset_flag = False

    def reset(self) -> None:
        self.current_state = self.coeff(2020)
        self.reset_flag = True

    def divergence(self, r, theta, phi, B_r, B_theta, B_phi):
        dphi = np.gradient(phi, axis=1)
        dtheta = np.gradient(theta, axis=0)
        dB_theta_sin_dtheta = np.gradient(B_theta*np.sin(theta), axis=0) / dtheta
        dB_phi_dphi = np.gradient(B_phi, axis=1) / dphi
        div = (1/(r*np.sin(theta))) * (dB_theta_sin_dtheta + dB_phi_dphi)
        return np.mean(np.abs(div))
        
    def field(self, lat: np.ndarray | float, lon: np.ndarray | float, coeff = None) -> List[np.ndarray | float]:
        if coeff is None:
            coeff = self.coeff
        radius = 6371.2
        theta = 90 - lat   # Co-latitude
        phi = lon
        B_r, B_theta, B_phi = synth_values(coeff, radius, theta, phi, grid = not isinstance(lat, float))
        F = (B_r**2 + B_theta**2 + B_phi**2)**0.5
        return F, B_r, B_theta, B_phi

    def contour_plot(self, lat_spacing: int, lon_spacing: int, coeff = None) -> List[np.ndarray]: # unpack in plt.contour
        if coeff is None:
            coeff = self.coeff
        lat = np.linspace(89, -89, lat_spacing)
        lon = np.linspace(-180, 180, lon_spacing)
        return np.meshgrid(lon, lat) + [self.field(lat, lon, coeff)]

    def step(self, year, action: np.ndarray) -> tuple:
        assert self.reset_flag, 'Stepping without resetting environment'
        init_state = self.current_state
        self.current_state = self.current_state + action
        reward = self.reward_fun(self.current_state, year, init_state, action)
        return self.current_state, reward

    def reward_fun(self, s_, year, s, a):

        LON, LAT, (F, B_r, B_theta, B_phi) = self.contour_plot(10,10,s_)
        div = self.divergence(6371.2, 90-LAT, LON, B_r, B_theta, B_phi)

        cont = torch.mean((s-s_)**2)

        
        # alpha = 100
        # beta = 0
        # gamma = 0
        # R1 = sum([(self.field(lat,lon,coeff=s_)[0] - F_obs)**2 for (lat,lon),(F_obs, F_g) in self.catalog(year).items()]) \
        #      / sum([(F_g - F_obs)**2 for (lat,lon),(F_obs, F_g) in self.catalog(year).items()]) if len(self.catalog(year))!=0 else 0
        # R2 = 1e-4*np.mean(abs(self.contour_plot(100,150,coeff=s_)[2] - self.contour_plot(100,150)[2]))
        # R3 = abs(sum(abs(s_[120:])-60))
        # return -alpha * R1 - beta * R2 - gamma * R3
        return 100/div + 10*cont
    

class Trajectory(MDP):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def get_traj(self, year, ep_len, policy):
        self.reset()
        traj = [[]]
        for i in range(ep_len):
            h = None
            for j in range(2020,year-5,-5):
                state = self.current_state
                action, hid = policy.sample(state.to(device), h.to(device) if h is not None else h)
                action = action.detach().cpu()
                hid = hid.detach().cpu()
                s,r = self.step(j,action)
                traj[i].append((state, action, r, s, self.coeff(j), h))
                h = hid
            traj.append([])
        return traj