import numpy as np
from wsn.models import Network
from wsn.fitness import fitness, FitnessParams

def test_fitness_range():
    net = Network.random_network(n_nodes=10, area_size=100.0, seed=0)
    fp = FitnessParams(rc=25.0)
    ch_indices = np.array([0, 3, 7])
    F, details = fitness(net, ch_indices, fp)
    assert 0.0 <= F <= 1.0
    for v in details.values():
        assert 0.0 <= v <= 1.0
