from easy_tpp.config_factory import SimulatorConfig

class Simulator :
    
    def __init__(self, simulator_config : SimulatorConfig) -> None:
            
        self.save_dir = simulator_config.get('save_dir', None)
        self.start_time = simulator_config.get('start_time')
        self.end_time = simulator_config.get('end_time')
        self.history_batch = simulator_config.get('history_batch', None)
        self.model = simulator_config.get('pretrained_model', None)
        

    def run(self) -> None:
        """
        Run the simulation process.
        """
        # Implement the simulation logic here
        model = self.model
        history_data = self.history_data_module
        start_time = self.start_time
        end_time = self.end_time
        data_loader = history_data.get_dataloader(split = 'test')
        
        for batch in data_loader:
            
            simulation = model.simulate(start_time = start_time, end_time = end_time, batch = batch)
        