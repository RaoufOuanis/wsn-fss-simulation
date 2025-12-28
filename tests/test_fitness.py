import numpy as np
from wsn.models import Network
from wsn.fitness import fitness, FitnessParams

def test_fitness_components_bounded():
    net = Network.random_network(n_nodes=10, area_size=100.0, seed=0)
    fp = FitnessParams(rc=25.0, lam=1.0)
    ch_indices = np.array([0, 3, 7])

    F, details = fitness(net, ch_indices, fp)

    # Base objective and components are bounded in [0,1] by construction.
    assert 0.0 <= details["F_base"] <= 1.0
    for k, v in details.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    # Regularized fitness is not clipped (paper Eq. 13).
    assert F >= 0.0
    assert F <= details["F_base"] + fp.lam * 1.0 + 1e-9
