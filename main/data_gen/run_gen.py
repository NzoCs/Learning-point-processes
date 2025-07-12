from easy_tpp.data.generation import HawkesSimulator, SelfCorrecting


def main():
    # Define parameters
    params = {
        "mu": [0.2, 0.2],
        "alpha": [[0.4, 0],[0, 8]],
        "beta": [[1, 0],[0, 20]]
    }
    
    mu = params["mu"]
    alpha = params["alpha"]
    beta = params["beta"]
    try :
      dim_process = len(mu)
    except TypeError:
      dim_process = 1
    
    generator = HawkesSimulator(
        mu=mu,
        alpha=alpha,    
        beta=beta,
        dim_process=dim_process,
        start_time=100,
        end_time=130
    )

    # generator = SelfCorrecting(
    #     dim_process=1,
    #     start_time=100,
    #     end_time=200
    # )
    
    generator.generate_and_save(output_dir='./data/test', num_simulations=10, splits={'train': 0.6, 'test': 0.2, 'dev': 0.2})

if __name__ == '__main__' :
    main()