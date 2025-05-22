import agents

# Configuration parameters
CONFIG = {
    'seed': None,
    'save_video': False,  # Boolean
    'num_agents': 50,
    'world_size': 20,
    'max_steps': 1000,
    'min_change_threshold': 0.2, # Minimum change DX (per agent) to continue simulation 
    'agent_strategy': 'rule_1',  #('rule_1', 'rule_2', 'random_movement')
    
    'agent_params': {
        'perception_radius': None,  # 'random' or float(5.0 default), None for inf perception
        'max_velocity': 0.4,     # 'random' or float  (0.4 default)
        
        # The following parameters remain constant for all experiments
        'velocity_gain': 1.4,
        'repulsion_strength': 0.05,
        'repulsion_radius': 0.5,
        'energy_loss': 0.95,
        'history_length': 30,
    }
}



agents.main(CONFIG)

