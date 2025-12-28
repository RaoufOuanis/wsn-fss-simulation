from wsn.models import Network
from wsn.energy import RadioParams, apply_round_energy

def test_energy_round_basic():
    net = Network.random_network(n_nodes=5, area_size=50.0, seed=0)
    radio = RadioParams()
    ch_indices = [0]
    assignments, _, _ = net.assign_clusters(ch_indices, rc=25.0)
    stats = apply_round_energy(net, ch_indices, assignments, radio)
    assert stats["alive"] <= 5
    assert stats["total_energy"] >= 0
