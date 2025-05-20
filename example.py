import agents

# Configuration parameters
CONFIG = {
    'seed': 0,
    'save_video': False,
    'num_agents': 50,
    'world_size': 20,
    'max_steps': 1000,
    'min_change_threshold': 0.05, # Minimum change DX (per agent) to continue simulation 
    'agent_strategy': 'rule_2',  #('rule_1', 'rule_2', 'random_movement')
    
    'agent_params': {
        'velocity_gain': 0.5,
        'max_velocity': 'random',
        'repulsion_strength': 0.3,
        'repulsion_radius': 1.0,
        'energy_loss': 0.05,
        'history_length': 20,
    }
}



agents.main()

