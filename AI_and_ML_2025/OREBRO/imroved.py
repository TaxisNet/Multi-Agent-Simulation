import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle
from typing import List, Tuple, Optional
from collections import deque


# Configuration parameters
CONFIG = {
    'num_agents': 50,
    'world_size': 20,
    'max_steps': 1000,
    'min_change_threshold': 0.05, # Minimum change DX (per agent) to continue simulation 
    'agent_strategy': 'rule_2',  #('rule_1', 'rule_2', 'random_movement')
    'collision_avoidance': 'field', #('reynolds', 'field')
    
    
    'agent_params': {
        'velocity_gain': 0.5,
        'max_velocity': 1.0,
        'perception_radius': 2.0,
        'repulsion_strength': 0.1,
        'repulsion_radius': 1.0,
        'energy_loss': 0.05,
        'history_length': 20,
    }
}

class Agent:
    """
    Agent class for a multi-agent simulation with movement strategies and collision avoidance.
    """
    def __init__(self, pos: np.ndarray, bounds: np.ndarray, K_v: float = 1.0, max_vel: float = 2.0, per_rad: Optional[float] = None):
        """
        Initialize an agent.
        
        Args:
            pos: Initial position as [x, y]
            bounds: World boundaries as [width, height]
            vel: Movement velocity
            per_rad: Perception radius
        """
        self.position = np.array(pos)
        self.color = np.random.choice(['red', 'green', 'blue'])
        self.perception_radius = per_rad if per_rad is not None else CONFIG['agent_params']['perception_radius']
        self.K_v = CONFIG['agent_params']['velocity_gain']
        self.max_velocity = CONFIG['agent_params']['max_velocity']
        # Movement parameters
        self.repulsion_strength = CONFIG['agent_params']['repulsion_strength']
        self.repulsion_radius = CONFIG['agent_params']['repulsion_radius']
        self.world_bounds = np.array(bounds)
        
        # Agent references for strategies
        self.agentA = None
        self.agentB = None

        # Collision avoidance method
        if CONFIG['collision_avoidance'] == 'reynolds':
            self.collision_avoidance = self.reynolds_collision_avoidance
        elif CONFIG['collision_avoidance'] == 'field':
            self.collision_avoidance = self.collision_avoidance_field
        else:
            # Default to field method
            self.collision_avoidance = self.collision_avoidance_field

        # Energy loss on collision with boundaries (between 0.01 and 1.0)
        self.energy_loss = np.clip(CONFIG['agent_params']['energy_loss'], 0.01, 1.0)
        
        # Movement history for visualization (fixed size)
        history_length = CONFIG['agent_params']['history_length']
        self.history = deque(maxlen=history_length)
        self.history.append(self.position.copy())


        # Initialize random direction vector (for strategy 3)
        random_vec = np.random.random(2) * 2 - 1  # Random direction
        self.random_direction = random_vec / (np.linalg.norm(random_vec) + 1e-6)  # Normalize
        
    def set_other_agents(self, A, B):
        """Set reference agents for movement strategies"""
        self.agentA = A
        self.agentB = B

    def follow_line_segment(self):
        """
        Strategy 1: Follow the line segment between agents A and B.
        Moves toward the closest point on the line segment between A and B.
        
        Returns:
            Movement vector
        """
        AB_vec = self.agentB.position - self.agentA.position
        PA_vec = self.agentA.position - self.position
        
        # Calculate normalized projection point
        norm_AB = np.dot(AB_vec, AB_vec) + 1e-6
        t = np.clip(np.dot(-PA_vec, AB_vec) / norm_AB, 0.005, 0.995)
        
        # Vector from current position to the closest point on the line segment
        PC_vec = self.agentA.position + t * AB_vec - self.position
        
        return PC_vec
    
    def follow_line_extension(self):
        """
        Strategy 2: Follow the extension of the line beyond agent B.
        Moves toward a point on the extended line beyond B.
        
        Returns:
            Movement vector
        """
        AB_vec = self.agentB.position - self.agentA.position
        PA_vec = self.agentA.position - self.position
        
        # Calculate projection point beyond the segment (t > 1)
        norm_AB = np.dot(AB_vec, AB_vec) + 1e-6
        t = np.clip(np.dot(-PA_vec, AB_vec) / norm_AB, 1.1, None)
        
        # Vector from current position to the extension point
        vec = self.agentA.position + t * AB_vec - self.position
        
        return vec
    
    def random_movement(self):
        """
        Strategy 3: Move in a random direction.
        
        Returns:
            Random movement vector
        """
        return self.random_direction
    
    def reynolds_collision_avoidance(self, all_agents):
        """
        Implements separation behavior from Reynolds' Boids model.
        
        Args:
            all_agents: List of all agents in the simulation

        Returns:
            Separation vector
        """
        separation_vector = np.zeros(2)
        neighbor_count = 0
        
        for other in all_agents:
            if other is self:
                continue
                
            direction = self.position - other.position
            distance = np.linalg.norm(direction)
            
            if distance < self.repulsion_radius and distance > 0:
                # Weight inversely by distance
                separation_vector += direction / distance**2
                neighbor_count += 1
        
        if neighbor_count > 0:
            separation_vector /= neighbor_count
            # Normalize to unit vector
            separation_vector = separation_vector / (np.linalg.norm(separation_vector) + 1e-6)
        
        return separation_vector * self.repulsion_strength
    
    def collision_avoidance_field(self, all_agents):
        """Add a repulsion vector to avoid collisions with other agents
        
        Args:
            all_agents: List of all agents in the simulation
        
        Returns:
            Repulsion vector
        """
        repulsion_vector = np.zeros(2)
        
        for other in all_agents:
            if other is self:
                continue
                
            # Vector from other to self
            direction = self.position - other.position
            distance = np.linalg.norm(direction)
            
            # If within collision radius, add repulsion force
            if distance < self.repulsion_radius:
                # Normalize and scale by inverse square of distance
                repulsion = direction / (distance**3+ 1e-6) 
                repulsion_vector += repulsion
        repulsion_vector = repulsion_vector / (np.linalg.norm(repulsion_vector) + 1e-6)

        return self.repulsion_strength * repulsion_vector
        

    def update_position(self, vector, all_agents):
        """
        Update agent position based on movement vector and collision avoidance.

        Args:
            vector: Base movement vector
            all_agents: List of all agents for collision avoidance
        """
        # Calculate velocity with collision avoidance
        vel = self.K_v * vector
        if self.collision_avoidance is not None:
            vel += self.collision_avoidance(all_agents)

        # Normalize velocity to max velocity 
        norm_vel = np.linalg.norm(vel)
        if norm_vel > self.max_velocity:
            vel = (vel / norm_vel) * self.max_velocity

        # Store original velocity for reflection calculations
        original_vel = vel.copy()
            
        # Proposed new position without boundary check
        unconstrained_position = self.position + vel
        
        # Check for boundary violations
        new_position = unconstrained_position.copy()

        # Handle boundary collisions properly
        # For each dimension check if we're crossing the boundary
        for i in range(len(self.world_bounds)):
            if abs(new_position[i]) > self.world_bounds[i]:
                # Determine how far we've gone past the boundary
                sign = 1 if new_position[i] > 0 else -1
                boundary = sign * self.world_bounds[i]
                
                # Calculate where along the path we hit the boundary
                # This is a fraction between 0 and 1
                fraction = (boundary - self.position[i]) / (unconstrained_position[i] - self.position[i])
                
                # Move to the boundary point
                new_position[i] = boundary
                
                # Calculate the remaining velocity after hitting the boundary
                remaining_vel = (1 - fraction) * original_vel[i]
                
                # Apply energy loss and reflection
                vel[i] = -remaining_vel * (1 - self.energy_loss)
                
                # Apply the reflection - move in the opposite direction with reduced energy
                new_position[i] += vel[i]

        # Update position
        self.position = new_position

        # Store position in history for visualization
        self.history.append(self.position.copy())
    
    def get_strategy_vector(self, strategy='follow_line_extension'):
        """
        Get movement vector based on selected strategy.
        
        Args:
            strategy: Strategy name ('follow_line_segment', 'follow_line_extension', 'random_movement')
            
        Returns:
            Movement vector
        """
        if strategy == 'rule_1':
            return self.follow_line_segment()
        elif strategy == 'rule_2':
            return self.follow_line_extension()
        elif strategy == 'random_movement':
            return self.random_movement()
        else:
            return self.follow_line_segment()  # Default


def random_position(world_limit):
    """Generate random position within world limits"""
    return 2 * world_limit * (np.random.random(2) - 0.5)


def visualize(agents, scatter, ax, step, show_trails=True, trail_length=10):
    """
    Update visualization of agents.
    
    Args:
        agents: List of agents
        scatter: Matplotlib scatter plot object
        ax: Matplotlib axis
        step: Current simulation step
        show_trails: Whether to show agent movement trails
        trail_length: Length of movement trails
    """
    # Update title and scatter positions
    ax.set_title(f"Step {step}")
    positions = [agent.position for agent in agents]
    scatter.set_offsets(positions)
    
    # Clear previous trails
    for artist in ax.lines:
        artist.remove()
    
    # Draw trails if enabled
    if show_trails:
        for agent in agents:
            # Get recent history points (limited by trail_length)
            history = [agent.history[i] for i in range(-trail_length,0)] if len(agent.history) > trail_length else agent.history
            if len(history) > 1:
                x_points = [p[0] for p in history]
                y_points = [p[1] for p in history]
                ax.plot(x_points, y_points, '-', color=agent.color, alpha=0.3, linewidth=1)
    
    # Draw perception circles for a few agents (for visualization)
    for artist in ax.patches:
        artist.remove()
    
    # Show perception radius for a few agents
    if len(agents) > 0:
        for i in range(min(3, len(agents))):
            circle = Circle(agents[i].position, agents[i].perception_radius, 
                           fill=False, linestyle='--', alpha=0.3, color=agents[i].color)
            ax.add_patch(circle)
    
    plt.pause(0.05)


def main():
    """Main simulation function"""
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Get configuration parameters
    N = CONFIG['num_agents']
    world_size = CONFIG['world_size']
    world_lim = np.array([world_size, world_size])
    
    # Create agents
    all_agents = [Agent(random_position(world_lim), world_lim, max_vel=CONFIG['agent_params']['max_velocity']) for _ in range(N)]
    
    # Assign reference agents for strategies
    for i in range(N):
        indx = list(range(N))
        indx.remove(i)
       
        A, B = np.random.choice(indx, 2, replace=False)
        all_agents[i].set_other_agents(all_agents[A], all_agents[B])
    
    # Simulation parameters
    step_limit = CONFIG['max_steps']
    minimum_change = CONFIG['min_change_threshold'] * N
    
    # Setup visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    delta = 2
    ax.set_xlim(-world_lim[0]-delta, world_lim[0]+delta)
    ax.set_ylim(-world_lim[1]-delta, world_lim[1]+delta)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create scatter plot
    scatter = ax.scatter([a.position[0] for a in all_agents], 
                        [a.position[1] for a in all_agents], 
                        c=[a.color for a in all_agents],
                        s=50)  # Larger points
    
    # Add legend
    ax.text(-world_lim[0]-0.5, world_lim[1]+1, 
           "Red/Green/Blue: Agent colors\nDashed circles: Agent perception radius", 
           fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Run simulation
    step, change = 0, float('inf')
    start_time = time.time()
    
    while step <= step_limit and change > minimum_change:
        # Visualize current state
        visualize(all_agents, scatter, ax, step, show_trails=True)
        
       
        change = 0
        
        # Update all agents
        for a in all_agents:
            vec = a.get_strategy_vector(CONFIG['agent_strategy'])
            a.update_position(vec, all_agents=all_agents)
            # Calculate total movement (for convergence check)
            change += np.linalg.norm(vec)
        
        step += 1
    # Display final state and statistics
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {step} steps")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Final movement magnitude: {change:.4f}")
    
    # Switch to interactive mode for final display
    plt.ioff()
    plt.suptitle("Agent Simulation - Final State", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()