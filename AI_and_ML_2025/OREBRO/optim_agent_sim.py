import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)

class Agent():
    def __init__(self, pos, bounds, vel=1, per_rad=None):
        self.position = np.array(pos)
        self.color = np.random.choice(['red', 'green', 'blue'])
        self.perception_radius = per_rad
        self.velocity = vel
        
        self.repulsion_strength = 5
        self.repulsion_radius = 0.5
        self.world_bounds = np.array(bounds)
        self.agentA = None
        self.agentB = None

        self.collision_avoidance = self.reynolds_collision_avoidance
        self.energy_loss = 0.05 
        self.energy_loss = np.clip(self.energy_loss, 0.01, 1)
    def set_other_agents(self, A, B):
        self.agentA = A
        self.agentB = B


    # Agent Update Law Strategies
    def strat1(self):
        AB_vec = self.agentB.position - self.agentA.position
        PA_vec = self.agentA.position - self.position
        # PB_vec = self.agentB.position - self.position
        
        norm_AB = np.dot(AB_vec, AB_vec) + 1e-6
        # t = np.clip(np.dot(-PA_vec, AB_vec) / norm_AB, 0, 1)
        t = np.clip(np.dot(-PA_vec, AB_vec) / norm_AB, 0.005, 0.995)
        PC_vec = self.agentA.position + t * AB_vec - self.position
        
        return PC_vec / (np.linalg.norm(PC_vec) + 1e-6) if np.linalg.norm(PC_vec) >= 1 else PC_vec
    
    def strat2(self):
        AB_vec = self.agentB.position - self.agentA.position
        PA_vec = self.agentA.position - self.position
        PB_vec = self.agentB.position - self.position
        
        norm_AB = np.dot(AB_vec, AB_vec) + 1e-6
        t = np.dot(-PA_vec, AB_vec) / norm_AB
        t = np.clip(np.dot(-PA_vec, AB_vec) / norm_AB, 1.1, None)
        
        vec = self.agentA.position + t * AB_vec - self.position
        
        return vec / (np.linalg.norm(vec) + 1e-6) if np.linalg.norm(vec) >= 1 else vec
    
    def strat3(self):
        ''' random vector '''
        
        return np.array((1,1), dtype=float) 

    #########################################################
    # Agent Collision Avoidance Methods

    def compute_repulsion(self, all_agents):
        repulsion_force = np.array([0.0, 0.0])
        for other in all_agents:
            if other is self:
                continue
            dist_vec = self.position - other.position
            dist_sq = np.dot(dist_vec, dist_vec)
            if dist_sq > self.repulsion_radius**2:
                continue  # Skip agents that are too far away
            
            repulsion_force += (dist_vec / (dist_sq + 1e-6)) * self.repulsion_strength
        return repulsion_force
    
    # def velocity_obstacle_avoidance(self, all_agents, time_horizon=2.0, agent_radius=0.5):
    #     """Adjust velocity to avoid collisions using velocity obstacles"""
    #     avoidance_vector = np.zeros(2)
        
    #     for other in all_agents:
    #         if other is self:
    #             continue
                
    #         relative_pos = other.position - self.position
    #         distance = np.linalg.norm(relative_pos)
            
    #         # Only consider nearby agents
    #         if distance > 5 * agent_radius:
    #             continue
                
    #         # Simplified velocity obstacle calculation
    #         # Calculate time to collision if any
    #         relative_pos_normalized = relative_pos / distance
    #         dot_product = np.dot(self.velocity * relative_pos_normalized, relative_pos_normalized)
            
    #         # If moving toward each other
    #         if dot_product > 0:
    #             # Calculate avoidance direction (perpendicular to relative position)
    #             perp_dir = np.array([-relative_pos[1], relative_pos[0]])
    #             perp_dir = perp_dir / np.linalg.norm(perp_dir)
                
    #             # Strength proportional to closeness and velocity
    #             strength = (1.0 / distance) * np.linalg.norm(self.velocity) * time_horizon
    #             avoidance_vector += perp_dir * strength
        
    #     return avoidance_vector
    
    def reynolds_collision_avoidance(self, all_agents, perception_radius=3.0, separation_weight=1.5):
        """Implements separation behavior from Reynolds' Boids model"""
        separation_vector = np.zeros(2)
        neighbor_count = 0
        
        for other in all_agents:
            if other is self:
                continue
                
            direction = self.position - other.position
            distance = np.linalg.norm(direction)
            
            if distance < perception_radius and distance > 0:
                # Weight inversely by distance
                separation_vector += direction / distance**2
                neighbor_count += 1
        
        if neighbor_count > 0:
            separation_vector /= neighbor_count
            # Normalize to unit vector
            separation_vector = separation_vector / np.linalg.norm(separation_vector+1e-6)
        
        return separation_vector * separation_weight


    def collision_avoidance_field(self, all_agents, collision_radius=1.0, repulsion_strength=0.5):
        """Add a repulsion vector to avoid collisions with other agents"""
        repulsion_vector = np.zeros(2)
        
        for other in all_agents:
            if other is self:
                continue
                
            # Vector from other to self
            direction = self.position - other.position
            distance = np.linalg.norm(direction)
            
            # If within collision radius, add repulsion force
            if distance < collision_radius and distance > 0:
                # Normalize and scale by inverse square of distance
                repulsion = direction / distance * (1/distance**2) * repulsion_strength
                repulsion_vector += repulsion
                
        return repulsion_vector



    def update_law(self, vector, all_agents):
        # Update position based on the vector
        vel =  self.velocity * vector
        if self.collision_avoidance is not None:
            vel += self.collision_avoidance(all_agents)
        new_position = self.position + vel
        
        # np.clip(self.position, -self.world_bounds, self.world_bounds, out=self.position)

        # Bounday collision
        if np.any(np.abs(new_position) > self.world_bounds):
            if (new_position > self.world_bounds).any():
                overshoot = new_position - self.world_bounds
            else:
                overshoot = new_position + self.world_bounds

            new_position = new_position - (2 - self.energy_loss)*overshoot 
            
        # Update position
        self.position = new_position
    
    def strat(self):
        return self.strat2()

def random_pos(world_limit):
    return 2 * world_limit * (np.random.random(2) - 0.5)

def visualize(agentArray, scatter, ax, step):
    ax.set_title(f"Step {step}")
    scatter.set_offsets([agent.position for agent in agentArray])
    plt.pause(0.1)

def main():
    N = 50
    diam = 50
    world_lim = np.array([diam, diam])
    all_agents = [Agent(random_pos(world_lim), world_lim) for _ in range(N)]
    
    for i in range(N):
        indx = list(range(N))
        indx.remove(i)
        A, B = random.sample(indx,2)
        all_agents[i].set_other_agents(all_agents[A], all_agents[B])
    
    step_limit = 1000
    minimum_change = 0.01*N
    
    plt.ion()
    fig, ax = plt.subplots()
    delta = 0.5
    ax.set_xlim(-world_lim[0]-delta, world_lim[0]+delta)
    ax.set_ylim(-world_lim[1]-delta, world_lim[1]+delta)
    scatter = ax.scatter([a.position[0] for a in all_agents], [a.position[1] for a in all_agents], c=[a.color for a in all_agents])
    
    step, change = 0, 1
    while step <= step_limit and change > minimum_change:
        visualize(all_agents, scatter, ax, step)
        
        change = sum(np.linalg.norm(a.strat()) for a in all_agents)
        # print(change)
        for a in all_agents:
            a.update_law(a.strat(), all_agents=all_agents)
        
        step += 1
    
    plt.ioff()
    plt.show()

main()