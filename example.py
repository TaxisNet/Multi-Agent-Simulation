import agents

# Configuration parameters
CONFIG = {
    'seed': 0,
    'save_video': True,
    'num_agents': 50,
    'world_size': 20,
    'max_steps': 500,
    'min_change_threshold': 0.05, # Minimum change DX (per agent) to continue simulation 
    'agent_strategy': 'rule_1',  #('rule_1', 'rule_2', 'random_movement')
    
    'agent_params': {
        'perception_radius': None,  # 'random' or float, None for inf perception
        'max_velocity': 0.4,  # 'random' or float
        
        # The following parameters remain constant for all experiments
        'velocity_gain': 0.2,
        'repulsion_strength': 0.05,
        'repulsion_radius': 0.5,
        'energy_loss': 0.05,
        'history_length': 30,
    }
}



agents.main(CONFIG)

