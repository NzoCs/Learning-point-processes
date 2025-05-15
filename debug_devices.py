import torch
from easy_tpp.config_factory import ModelConfig, SimulatorConfig
from easy_tpp.models import BaseModel
from easy_tpp.utils import logger

def check_tensor_devices(model):
    """Check device consistency for all parameters in a model."""
    devices = set()
    for name, param in model.named_parameters():
        devices.add(param.device)
        print(f"Parameter '{name}' is on device: {param.device}")
    
    if len(devices) > 1:
        print(f"WARNING: Model has parameters on different devices: {devices}")
    else:
        print(f"All parameters are on device: {list(devices)[0]}")
    
    return devices

def run_device_check():
    """Check if model parameters and tensors are on consistent devices."""
    # Create a minimal model config
    model_config = ModelConfig(
        model_id="NHP",  # Test with a simple model
        num_event_types=3,
        num_event_types_pad=4,
        pad_token_id=3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        specs={
            "hidden_size": 16,
            "loss_integral_num_sample_per_step": 5,
            "rnn_type": "LSTM",
            "use_ln": False
        },
        base_config={
            "lr": 0.001,
            "max_epochs": 10,
            "dropout": 0.1,
        },
        thinning={
            "num_sample": 10,
            "num_exp": 50,
            "over_sample_rate": 2.0,
            "num_samples_boundary": 5,
            "dtime_max": 5.0,
            "num_steps": 1
        },
        simulation_config={
            "batch_size": 2,
            "start_time": 0.0,
            "end_time": 10.0
        }
    )
    
    # Create a model
    print("Creating model...")
    model = BaseModel.generate_model_from_config(model_config)
    
    # Check device consistency
    print("\nChecking model device consistency...")
    devices = check_tensor_devices(model)
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    try:
        batch_size = 2
        seq_len = 5
        
        # Create dummy batch data
        time_seqs = torch.cumsum(torch.ones(batch_size, seq_len), dim=1).to(model.device)
        time_delta_seqs = torch.ones(batch_size, seq_len).to(model.device)
        type_seqs = torch.randint(0, model.num_event_types, (batch_size, seq_len)).to(model.device)
        batch_non_pad_mask = torch.ones(batch_size, seq_len).to(model.device)
        attention_mask = torch.zeros(batch_size, seq_len, seq_len).to(model.device)
        
        batch = (time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask)
        
        print("Running loglike_loss...")
        loss, num_events = model.loglike_loss(batch)
        print(f"Loss computed: {loss.item()}, Num events: {num_events}")
        
        print("\nTesting simulation...")
        time_seq, time_delta_seq, event_seq, simul_mask = model.simulate(batch=batch)
        print(f"Simulation produced: {simul_mask.sum().item()} events")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_device_check()
