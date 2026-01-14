
from new_ltpp.data.generation import HawkesSimulator, SelfCorrecting


def generate_hawkes_data() -> None:
    """Generate synthetic Hawkes process data."""
    # Define Hawkes parameters
    params = {
        "mu": [0.2, 0.2],
        "alpha": [[0.4, 0], [0, 0.8]],
        "beta": [[1, 0], [0, 20]],
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
        start_time=0,
        end_time=100,
    )

    generator.generate_and_save(
        output_dir="./data/hawkes_synthetic",
        num_simulations=50,
        splits={"train": 0.6, "test": 0.2, "dev": 0.2},
    )
    print("Hawkes synthetic data generated in ./data/hawkes_synthetic")


def generate_self_correcting_data() -> None:
    """Generate synthetic Self-Correcting process data."""
    generator = SelfCorrecting(dim_process=1, start_time=0, end_time=200)

    generator.generate_and_save(
        output_dir="./data/self_correcting_synthetic",
        num_simulations=30,
        splits={"train": 0.7, "test": 0.15, "dev": 0.15},
    )
    print(
        "Self-Correcting synthetic data generated in ./data/self_correcting_synthetic"
    )


def generate_multivariate_hawkes() -> None:
    """Generate multivariate Hawkes process data."""
    # More complex multivariate parameters
    params = {
        "mu": [0.1, 0.15, 0.2],
        "alpha": [[0.3, 0.1, 0.05], [0.2, 0.4, 0.1], [0.1, 0.15, 0.35]],
        "beta": [[2, 1, 0.5], [1.5, 3, 1], [1, 2, 2.5]],
    }

    generator = HawkesSimulator(
        mu=params["mu"],
        alpha=params["alpha"],
        beta=params["beta"],
        dim_process=3,
        start_time=0,
        end_time=150,
    )

    generator.generate_and_save(
        output_dir="./data/multivariate_hawkes",
        num_simulations=20,
        splits={"train": 0.6, "test": 0.25, "dev": 0.15},
    )
    print("Multivariate Hawkes data generated in ./data/multivariate_hawkes")


def main() -> None:
    generate_hawkes_data()
    generate_self_correcting_data()
    generate_multivariate_hawkes()


if __name__ == "__main__":
    main()
