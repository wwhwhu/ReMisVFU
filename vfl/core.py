from typing import List
import torch
from model.model import FirstNet, SecondNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Client(nn.Module):
    """
    Wrapper for Party A's feature extractor.
    """
    def __init__(self, in_channels=1, model_name=None):
        super(Client, self).__init__()
        if model_name is None:
            print("Using default model: FirstNet")
            self.part = FirstNet(in_channels)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    def forward(self, x):
        return self.part(x)


class Server(nn.Module):
    """
    Wrapper for server-side classifier.
    """
    def __init__(self, output_dim=10, num_parties=2, in_channels=1):
        super(Server, self).__init__()
        self.partC = SecondNet(output_dim, num_parties, in_channels)

    def forward(self, x_list):
        return self.partC(x_list)

class SplitNN(nn.Module):
    """
    Split Neural Network coordinating PartyA, PartyB, and Server.
    - clientA, clientB, server: nn.Module instances
    - clientA_optimizer, clientB_optimizer, server_optimizer: optimizers for each part
    """
    def __init__(self, num_parties, in_channel=1, out_dim=10):
        super(SplitNN, self).__init__()
        self.client_list = nn.ModuleList([Client(in_channel) for _ in range(num_parties)])
        self.client_opt_list = [optim.Adam(c.parameters(), lr=1e-3)
                            for c in self.client_list]
        self.server = Server(output_dim=out_dim, num_parties=num_parties, in_channels=in_channel)
        self.server_opt = optim.Adam(self.server.parameters(), lr=0.001)

    def forward(self, xs: List[torch.Tensor]):
        # Party A and Party B compute their feature maps
        feats = [client(x) for client, x in zip(self.client_list, xs)]
        # Server computes final logits
        out = self.server(feats)
        return out

    def do_zero_grads(self):
        """
        Zero gradients for all three networks.
        """
        for client_opt in self.client_opt_list:
            client_opt.zero_grad()
        self.server_opt.zero_grad()

    def doStep(self):
        """
        Step optimizers for all three networks.
        """
        for client_opt in self.client_opt_list:
            client_opt.step()
        self.server_opt.step()
    
    def toDevice(self, device):
        """
        Move all parts of the SplitNN to the specified device.
        """
        for client in self.client_list:
            client.to(device)
        self.server.to(device)
        print(f"SplitNN moved to device: {device}")

class Fusion:
    """
    Base class for model parameter fusion (aggregation).
    """
    def __init__(self, num_parties):
        self.num_parties = num_parties

    def average_selected_models(self, selected_party_states):
        """
        Given a list of state_dicts for selected parties, compute element-wise average.
        Returns a new state_dict with averaged parameters.
        """
        num = len(selected_party_states)
        avg_state = {}
        for key in selected_party_states[0].keys():
            # Initialize sum with zeros
            avg_state[key] = torch.zeros_like(selected_party_states[0][key])
            # Sum over all selected states
            for s in selected_party_states:
                avg_state[key] += s[key]
            # Divide by number of selected
            avg_state[key] /= num
        return avg_state

    def fusion_algo(self, party_state_dicts):
        """
        To be implemented by subclasses.
        party_state_dicts: list of full SplitNN state_dicts for each party (all parts).
        """
        raise NotImplementedError
    
class FusionAvg(Fusion):
    """
    Simple FedAvg: average all parties' full state_dicts.
    """
    def __init__(self, num_parties):
        super(FusionAvg, self).__init__(num_parties)

    def fusion_algo(self, party_state_dicts):
        return self.average_selected_models(party_state_dicts)


def FL_round_fusion_selection(num_parties, fusion_key='FedAvg'):
    """
    Factory method: returns a Fusion instance based on fusion_key.
    """
    if fusion_key == 'FedAvg':
        return FusionAvg(num_parties)
    else:
        raise ValueError(f"Unsupported fusion_key: {fusion_key}")