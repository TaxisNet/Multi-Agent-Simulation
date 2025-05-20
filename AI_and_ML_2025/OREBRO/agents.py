import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FFMpegWriter 

import time
import datetime
from typing import Optional
from collections import deque


# Configuration parameters
CONFIG = {
    'seed': None,
    'save_video': False,
    'num_agents': 50,
    'world_size': 20,
    'max_steps': 600,
    'min_change_threshold':  0.1, # Minimum change DX (per agent) to continue simulation 
    'agent_strategy': 'rule_2',  #('rule_1', 'rule_2', 'random_movement')
    
    'agent_params': {
        'perception_radius': None,
        'velocity_gain': 0.6,
        'max_velocity': 1.0 ,  # 'random' or float
        'repulsion_strength': 0.2,
        'repulsion_radius': 1.0,
        'energy_loss': 0.05,
        'history_length': 20,
    }
}

class Agent:
    """
    Agent class for a multi-agent simulation with movement strategies and collision avoidance.
    """
    def __init__(self, pos: np.ndarray, bounds: np.ndarray, K_v: float = 1.0, max_vel: float = 2.0, max_velocity: float = None, repulsion_radius: Optional[float] = None, repulsion_strength: Optional[float] = None, perception_radius: Optional[float] = None):
        """
        Initialize an agent.
        
        Args:
            pos: Initial position as [x, y]
            bounds: World boundaries as [width, height]
            vel: Movement velocity
            per_rad: Perception radius
        """
        self.position = np.array(pos)
        self.perception_radius = perception_radius
        # self.color = np.random.choice(['red', 'green', 'blue', ])
        self.color = np.random.choice(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
        # Controler parameters
        self.K_v = K_v
        self.max_velocity = max_vel
        # Movement parameters
        self.repulsion_strength = repulsion_strength
        self.repulsion_radius = repulsion_radius
        self.world_bounds = np.array(bounds)
        
        # Agent references for strategies
        self.agentA = None
        self.agentB = None

        # Waypooint for random movement
        self.waypoint = None

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

    def perception_logic(self):
        # Preception based logic
        if self.perception_radius is not None:
            # Check if within perception radius
            distance_to_A = np.linalg.norm(self.position - self.agentA.position)
            distance_to_B = np.linalg.norm(self.position - self.agentB.position)

            # If you can't see either agent, go to random waypoint
            if distance_to_A > self.perception_radius and distance_to_B > self.perception_radius:
                return self.random_movement()
            # If you can see one agent, move towards it
            elif distance_to_A < self.perception_radius and distance_to_B > self.perception_radius:
                return self.agentA.position - self.position
            elif distance_to_A > self.perception_radius and distance_to_B < self.perception_radius:
                return self.agentB.position - self.position
            
            return None
        
    def follow_line_segment(self):
        """
        Strategy 1: Follow the line segment between agents A and B.
        Moves toward the closest point on the line segment between A and B.
        
        Returns:
            Movement vector
        """
        # Do perception based logic rule, incase you cant see both agents
        perception_logic_vector = self.perception_logic()
        if perception_logic_vector is not None:
            return perception_logic_vector
            

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

        # Do perception based logic rule, incase you cant see both agents
        perception_logic_vector = self.perception_logic()
        if perception_logic_vector is not None:
            return perception_logic_vector
            

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
        if self.waypoint is None:
            # Generate a random waypoint within the world bounds
            self.waypoint = random_position(self.world_bounds)
            
        vec_to_waypoint = self.position - self.waypoint
        distance_to_waypoint = np.linalg.norm(vec_to_waypoint)
        
        if distance_to_waypoint < 0.1:
            # If close to waypoint, generate a new one
            self.waypoint = random_position(self.world_bounds)
            vec_to_waypoint = self.position - self.waypoint
        return vec_to_waypoint
    
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
    if len(agents) > 0 and agents[0].perception_radius is not None:
        for i in range(min(3, len(agents))):
            circle = Circle(agents[i].position, agents[i].perception_radius, 
                           fill=False, linestyle='--', alpha=0.3, color=agents[i].color)
            ax.add_patch(circle)
    
    plt.pause(0.05)


def main():
    """Main simulation function"""
    # Set random seed for reproducibility
    np.random.seed(CONFIG['seed'])
    
    # Get configuration parameters
    N = CONFIG['num_agents']
    world_size = CONFIG['world_size']
    world_lim = np.array([world_size, world_size])

    
        
    
    # Create agents
    all_agents = [Agent(random_position(world_lim), 
                        bounds = world_lim,
                        perception_radius = CONFIG['agent_params']['perception_radius'],
                        K_v=CONFIG['agent_params']['velocity_gain'], 
                        max_vel= np.random.uniform(0.1, 4.0) if CONFIG['agent_params']['max_velocity'] == 'random' else CONFIG['agent_params']['max_velocity'],
                        repulsion_strength = CONFIG['agent_params']['repulsion_strength'],
                        repulsion_radius = CONFIG['agent_params']['repulsion_radius'],
                        )
                        for _ in range(N)]
    
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
    delta = 1
    ax.set_xlim(-world_lim[0]-delta, world_lim[0]+delta)
    ax.set_ylim(-world_lim[1]-delta, world_lim[1]+delta)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Draw bold boundary lines at world limits
    for x in [-world_lim[0], world_lim[0]]:
        ax.plot([x, x], [-world_lim[1], world_lim[1]], color='blue', linewidth=2.5)
    for y in [-world_lim[1], world_lim[1]]:
        ax.plot([-world_lim[0], world_lim[0]], [y, y], color='black', linewidth=2.5)
    
    # Create scatter plot
    scatter = ax.scatter([a.position[0] for a in all_agents], 
                        [a.position[1] for a in all_agents], 
                        c=[a.color for a in all_agents],
                        s=50)  # Larger points
    
    # Add legend
    ax.text(-world_lim[0]-0.5, world_lim[1]+1, 
           "Colored Dots: Agent colors\nDashed circles: Agent Perception Radius", 
           fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Video saver initialization
    if CONFIG['save_video']:
        # Set up video writer
        video_filename = f"multiagent_simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = FFMpegWriter(fps=20, metadata=dict(title='Agent Simulation', artist='Matplotlib'), 
                              bitrate=1800)
        writer.setup(fig, video_filename, dpi=100)

    # Run simulation Loop
    step, change = 0, float('inf')
    start_time = time.time()
    
    while step <= step_limit and change > minimum_change:
        # Visualize current state
        visualize(all_agents, scatter, ax, step, show_trails=True)

        if CONFIG['save_video']:
            writer.grab_frame()
       
        change = 0
        
        # Update all agents
        for a in all_agents:
            vec = a.get_strategy_vector(CONFIG['agent_strategy'])
            a.update_position(vec, all_agents=all_agents)
            # Calculate total movement (for convergence check)
            change += np.linalg.norm(vec)
        
        step += 1

    if CONFIG['save_video']:
        writer.finish()
        print(f"Video saved as {video_filename}")
    # Display final state and statistics
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {step} steps")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Final movement magnitude: {change:.4f}")
    
    


if __name__ == "__main__":
    main()