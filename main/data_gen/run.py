import argparse

from easy_tpp.config_factory import Config
from easy_tpp.data_gen import BaseSimulator, HawkesSimulator, SelfCorrecting


def main():
    # Define parameters
    params = {
        "mu": [0.2, 0.2],
        "alpha": [[0.4, 0], [0, 0.4]],
        "beta": [[1.0, 0], [0, 20]]
    }
    
    mu = params["mu"]
    alpha = params["alpha"]
    beta = params["beta"]
    dim_process = len(mu)
    
    generator = HawkesSimulator(
        mu=mu,
        alpha=alpha,    
        beta=beta,
        dim_process=dim_process,
        start_time=100,
        end_time=200
    )

    # generator = SelfCorrecting(
    #     mu=mu,
    #     alpha=alpha,
    #     dim_process=1,
    #     start_time=100,
    #     end_time=200
    # )
    
    generator.generate_and_save(output_dir='../results/data/syn/test_dataset', num_simulations=1000, splits={'train': 0.6, 'test': 0.2, 'dev': 0.2})
    
if __name__ == '__main__' : 
    main()