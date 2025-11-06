import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import time

@dataclass
class Shell:
    """Represents a shell with physical properties"""
    name: str
    mass: float  # kg
    caliber: float  # mm
    muzzle_velocity: float  # m/s
    drag_coefficient: float  # dimensionless
    cross_sectional_area: float  # m²

@dataclass
class Gun:
    """Represents a gun with firing characteristics"""
    name: str
    max_elevation: float  # degrees
    min_elevation: float  # degrees
    barrel_length: float  # m

@dataclass
class Target:
    """Represents a target to hit"""
    distance: float  # m (north of firing position)
    elevation: float  # m (height above ground)
    size: float  # m (target size for hit detection)

class ProjectileSimulator:
    """Simulates projectile motion with air resistance"""
    
    def __init__(self):
        # Physics constants
        self.g = 9.81  # m/s² (gravity)
        self.rho_air = 1.225  # kg/m³ (air density at sea level)
        
        # Predefined shells
        self.shells = {
            "105mm HE": Shell("105mm HE", 15.0, 105, 650, 0.3, math.pi * (0.105/2)**2),
            "155mm HE": Shell("155mm HE", 43.5, 155, 827, 0.35, math.pi * (0.155/2)**2),
            "120mm APFSDS": Shell("120mm APFSDS", 8.1, 120, 1700, 0.22, math.pi * (0.120/2)**2),
            "81mm Mortar": Shell("81mm Mortar", 4.2, 81, 300, 0.4, math.pi * (0.081/2)**2)
        }
        
        # Predefined guns
        self.guns = {
            "M101 Howitzer": Gun("M101 Howitzer", 65, -5, 2.3),
            "M198 Howitzer": Gun("M198 Howitzer", 72, -5, 6.1),
            "M1A2 Tank": Gun("M1A2 Tank", 20, -10, 6.4),
            "81mm Mortar": Gun("81mm Mortar", 85, 45, 1.2)
        }
    
    def calculate_trajectory(self, shell: Shell, gun: Gun, elevation_angle: float, 
                           initial_velocity: float = None, time_step: float = 0.01) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate projectile trajectory
        
        Args:
            shell: Shell object with physical properties
            gun: Gun object with firing characteristics
            elevation_angle: Firing angle in degrees (0 = horizontal, 90 = vertical)
            initial_velocity: Initial velocity (if None, uses shell's muzzle velocity)
            time_step: Time step for simulation in seconds
            
        Returns:
            Tuple of (x_positions, y_positions, times)
        """
        if initial_velocity is None:
            initial_velocity = shell.muzzle_velocity
        
        # Convert angle to radians
        angle_rad = math.radians(elevation_angle)
        
        # Initial velocity components
        vx0 = initial_velocity * math.cos(angle_rad)
        vy0 = initial_velocity * math.sin(angle_rad)
        
        # Initialize arrays
        x_positions = [0.0]
        y_positions = [0.0]
        times = [0.0]
        
        # Current state
        x, y = 0.0, 0.0
        vx, vy = vx0, vy0
        t = 0.0
        
        # Simulation loop
        while y >= 0:  # Continue until projectile hits ground
            # Calculate air resistance force
            velocity_magnitude = math.sqrt(vx**2 + vy**2)
            drag_force = 0.5 * self.rho_air * velocity_magnitude**2 * shell.drag_coefficient * shell.cross_sectional_area
            
            # Drag force components (opposite to velocity)
            if velocity_magnitude > 0:
                drag_x = -drag_force * vx / velocity_magnitude
                drag_y = -drag_force * vy / velocity_magnitude
            else:
                drag_x = drag_y = 0
            
            # Acceleration components
            ax = drag_x / shell.mass
            ay = -self.g + drag_y / shell.mass
            
            # Update velocity (Euler integration)
            vx += ax * time_step
            vy += ay * time_step
            
            # Update position
            x += vx * time_step
            y += vy * time_step
            
            # Update time
            t += time_step
            
            # Store positions
            x_positions.append(x)
            y_positions.append(y)
            times.append(t)
            
            # Safety check to prevent infinite loops
            if t > 1000:  # 1000 seconds max
                break
            
            # Debug: Check if we're making progress
            if len(x_positions) > 1000 and x < 1.0:  # If we've taken 1000 steps and moved less than 1m
                print(f"Warning: Trajectory calculation may be stuck at angle {elevation_angle}°")
                break
        
        # If we didn't hit the ground, add one more point to show where we ended up
        if y > 0:
            x_positions.append(x)
            y_positions.append(y)
            times.append(t)
        
        return x_positions, y_positions, times
    
    def find_optimal_angle(self, shell: Shell, gun: Gun, target: Target, 
                          angle_step: float = 0.01) -> Tuple[float, float, float]:
        """
        Find the optimal firing angle to hit a target
        
        Returns:
            Tuple of (optimal_angle, range_at_angle, time_of_flight)
        """
        best_angle = None
        best_range = float('inf')
        best_time = 0
        best_miss = float('inf')
        
        # Try different angles with smaller step for precision
        for angle in np.arange(gun.min_elevation, gun.max_elevation + angle_step, angle_step):
            x_pos, y_pos, times = self.calculate_trajectory(shell, gun, angle)
            
            # Find where projectile hits ground (skip the initial position)
            for i in range(1, len(y_pos)):
                if y_pos[i] <= 0:
                    range_at_angle = x_pos[i]
                    time_of_flight = times[i]
                    
                    # Check if this angle gets closer to target
                    current_miss = abs(range_at_angle - target.distance)
                    
                    if current_miss < best_miss:
                        best_range = range_at_angle
                        best_angle = angle
                        best_time = time_of_flight
                        best_miss = current_miss
                        
                        # If we hit exactly, we can stop searching
                        if current_miss < 0.01:  # Within 1cm
                            return best_angle, best_range, best_time
                    break
        
        return best_angle, best_range, best_time
    
    def find_exact_hit_angle(self, shell: Shell, gun: Gun, target: Target, 
                            tolerance: float = 0.1) -> Tuple[float, float, float]:
        """
        Find the exact firing angle to hit a target within specified tolerance
        
        Args:
            shell: Shell object
            gun: Gun object  
            target: Target object
            tolerance: Acceptable miss distance in meters (default 0.1m = 10cm)
            
        Returns:
            Tuple of (exact_angle, range_at_angle, time_of_flight)
        """
        # First find a rough estimate
        rough_angle, rough_range, rough_time = self.find_optimal_angle(shell, gun, target, 0.1)
        
        if rough_angle is None:
            return None, 0, 0
        
        # Now do a fine search around the rough angle
        fine_step = 0.001  # 0.001 degree precision
        search_range = 0.5  # Search ±0.5 degrees around rough angle
        
        best_angle = rough_angle
        best_range = rough_range
        best_time = rough_time
        best_miss = abs(rough_range - target.distance)
        
        # Fine search
        for angle in np.arange(max(gun.min_elevation, rough_angle - search_range), 
                              min(gun.max_elevation, rough_angle + search_range + fine_step), fine_step):
            x_pos, y_pos, times = self.calculate_trajectory(shell, gun, angle)
            
            # Find where projectile hits ground
            for i in range(1, len(y_pos)):
                if y_pos[i] <= 0:
                    range_at_angle = x_pos[i]
                    time_of_flight = times[i]
                    current_miss = abs(range_at_angle - target.distance)
                    
                    if current_miss < best_miss:
                        best_range = range_at_angle
                        best_angle = angle
                        best_time = time_of_flight
                        best_miss = current_miss
                        
                        # If we're within tolerance, we can stop
                        if current_miss <= tolerance:
                            return best_angle, best_range, best_time
                    break
        
        return best_angle, best_range, best_time
    
    def simulate_hit(self, shell: Shell, gun: Gun, target: Target, 
                    elevation_angle: float, wind_speed: float = 0, wind_direction: float = 0) -> dict:
        """
        Simulate a shot and determine if it hits the target
        
        Returns:
            Dictionary with simulation results
        """
        x_pos, y_pos, times = self.calculate_trajectory(shell, gun, elevation_angle)
        
        # Find impact point (skip the initial position)
        impact_x = 0
        impact_y = 0
        impact_time = 0
        
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                impact_y = y_pos[i]
                impact_time = times[i]
                break
        
        # Calculate miss distance
        miss_distance = abs(impact_x - target.distance)
        
        # Determine if hit (considering target size)
        hit = miss_distance <= target.size / 2
        
        # Calculate accuracy metrics
        accuracy = max(0, 100 - (miss_distance / target.distance) * 100) if target.distance > 0 else 0
        
        return {
            'hit': hit,
            'impact_x': impact_x,
            'impact_y': impact_y,
            'impact_time': impact_time,
            'miss_distance': miss_distance,
            'accuracy': accuracy,
            'trajectory': (x_pos, y_pos, times)
        }
    
    def plot_trajectory(self, shell: Shell, gun: Gun, target: Target, 
                       elevation_angle: float, show_target: bool = True):
        """Plot the projectile trajectory"""
        x_pos, y_pos, times = self.calculate_trajectory(shell, gun, elevation_angle)
        
        plt.figure(figsize=(12, 8))
        
        # Plot trajectory
        plt.plot(x_pos, y_pos, 'b-', linewidth=2, label=f'{shell.name} trajectory')
        
        # Plot ground
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Plot target
        if show_target:
            plt.axvline(x=target.distance, color='r', linestyle='--', alpha=0.7, label='Target')
            plt.plot(target.distance, target.elevation, 'ro', markersize=10, label='Target position')
        
        # Plot firing position
        plt.plot(0, 0, 'ko', markersize=8, label='Firing position')
        
        # Calculate and display range
        impact_x = x_pos[-1] if y_pos[-1] <= 0 else x_pos[-2]
        plt.axvline(x=impact_x, color='g', linestyle=':', alpha=0.7, label=f'Impact at {impact_x:.1f}m')
        
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.title(f'Projectile Trajectory: {shell.name} from {gun.name}\nElevation: {elevation_angle}°')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(0, max(target.distance * 1.2, impact_x * 1.1))
        plt.ylim(0, max(max(y_pos) * 1.1, target.elevation * 1.2))
        plt.show()
    
    def interactive_simulation(self):
        """Run an interactive simulation with user input"""
        print("=== PROJECTILE MOTION SIMULATOR ===\n")
        
        # Select shell
        print("Available shells:")
        for i, (name, shell) in enumerate(self.shells.items(), 1):
            print(f"{i}. {name} (Mass: {shell.mass}kg, Velocity: {shell.muzzle_velocity}m/s)")
        
        shell_choice = int(input("\nSelect shell (1-4): ")) - 1
        shell_names = list(self.shells.keys())
        selected_shell = self.shells[shell_names[shell_choice]]
        
        # Select gun
        print("\nAvailable guns:")
        for i, (name, gun) in enumerate(self.guns.items(), 1):
            print(f"{i}. {name} (Elevation: {gun.min_elevation}° to {gun.max_elevation}°)")
        
        gun_choice = int(input("\nSelect gun (1-4): ")) - 1
        gun_names = list(self.guns.keys())
        selected_gun = self.guns[gun_names[gun_choice]]
        
        # Target parameters
        target_distance = float(input(f"\nEnter target distance (m) [default: 5000]: ") or 5000)
        target_elevation = float(input(f"Enter target elevation (m) [default: 0]: ") or 0)
        target_size = float(input(f"Enter target size (m) [default: 10]: ") or 10)
        
        target = Target(target_distance, target_elevation, target_size)
        
        # Find optimal angle
        print(f"\nCalculating optimal firing angle for target at {target_distance}m...")
        print(f"Testing angles from {selected_gun.min_elevation}° to {selected_gun.max_elevation}° in 0.01° increments...")
        
        # First find rough optimal angle
        optimal_angle, optimal_range, optimal_time = self.find_optimal_angle(selected_shell, selected_gun, target)
        
        if optimal_angle is not None:
            print(f"Rough optimal angle: {optimal_angle:.3f}°")
            print(f"Range at rough angle: {optimal_range:.1f}m")
            
            # Now find exact hit angle
            print(f"\nFinding exact hit angle...")
            exact_angle, exact_range, exact_time = self.find_exact_hit_angle(selected_shell, selected_gun, target, tolerance=0.1)
            
            if exact_angle is not None:
                optimal_angle = exact_angle
                optimal_range = exact_range
                optimal_time = exact_time
                print(f"Exact hit angle: {exact_angle:.4f}°")
                print(f"Range at exact angle: {exact_range:.3f}m")
                print(f"Miss distance: {abs(exact_range - target_distance):.3f}m")
        
        if optimal_angle is not None:
            print(f"Optimal firing angle: {optimal_angle:.1f}°")
            print(f"Range at optimal angle: {optimal_range:.1f}m")
            print(f"Time of flight: {optimal_time:.1f}s")
            
            # Simulate optimal shot
            result = self.simulate_hit(selected_shell, selected_gun, target, optimal_angle)
            
            print(f"\nSimulation Results:")
            print(f"Hit: {'Yes' if result['hit'] else 'No'}")
            print(f"Impact position: {result['impact_x']:.1f}m")
            print(f"Miss distance: {result['miss_distance']:.1f}m")
            print(f"Accuracy: {result['accuracy']:.1f}%")
            
            # Plot trajectory
            self.plot_trajectory(selected_shell, selected_gun, target, optimal_angle)
            
            # Try different angles
            print(f"\nTrying different angles...")
            angles_to_try = [optimal_angle - 2, optimal_angle - 1, optimal_angle, optimal_angle + 1, optimal_angle + 2]
            
            for angle in angles_to_try:
                if gun.min_elevation <= angle <= gun.max_elevation:
                    result = self.simulate_hit(selected_shell, selected_gun, target, angle)
                    print(f"Angle {angle:5.1f}°: {'HIT' if result['hit'] else 'MISS'} "
                          f"(Miss: {result['miss_distance']:6.1f}m, Accuracy: {result['accuracy']:5.1f}%)")
        else:
            print("No valid firing angle found for this target distance.")
    
    def batch_simulation(self, shell: Shell, gun: Gun, target: Target, 
                        angle_range: Tuple[float, float] = (0, 90), angle_step: float = 1.0):
        """Run batch simulation across a range of angles"""
        angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
        results = []
        
        print(f"Running batch simulation for {shell.name} from {gun.name}")
        print(f"Target: {target.distance}m away, {target.elevation}m elevation")
        print(f"Testing angles from {angle_range[0]}° to {angle_range[1]}°\n")
        
        for angle in angles:
            if gun.min_elevation <= angle <= gun.max_elevation:
                result = self.simulate_hit(shell, gun, target, angle)
                results.append((angle, result))
                
                status = "HIT" if result['hit'] else "MISS"
                print(f"Angle {angle:5.1f}°: {status} "
                      f"(Miss: {result['miss_distance']:6.1f}m, Accuracy: {result['accuracy']:5.1f}%)")
        
        # Find best angle
        if results:
            best_result = min(results, key=lambda x: x[1]['miss_distance'])
            print(f"\nBest angle: {best_result[0]:.1f}° with accuracy {best_result[1]['accuracy']:.1f}%")
            
            # Plot best trajectory
            self.plot_trajectory(shell, gun, target, best_result[0])
        
        return results

def main():
    """Main function to run the simulation"""
    simulator = ProjectileSimulator()
    
    print("Welcome to the Projectile Motion Simulator!")
    print("This simulator helps you find the optimal firing angle to hit targets.")
    print("Features:")
    print("- Multiple shell types (HE, AP, Mortar)")
    print("- Different gun characteristics")
    print("- Realistic physics with air resistance")
    print("- Target hit detection")
    print("- Trajectory visualization")
    
    while True:
        print("\n" + "="*50)
        print("Choose an option:")
        print("1. Interactive simulation")
        print("2. Quick demo")
        print("3. Batch simulation")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            simulator.interactive_simulation()
        elif choice == '2':
            # Quick demo
            print("\nRunning quick demo...")
            shell = simulator.shells["155mm HE"]
            gun = simulator.guns["M198 Howitzer"]
            target = Target(8000, 0, 15)
            
            print(f"Demo: {shell.name} from {gun.name} at target {target.distance}m away")
            
            optimal_angle, optimal_range, optimal_time = simulator.find_optimal_angle(shell, gun, target)
            print(f"Optimal angle: {optimal_angle:.1f}°")
            
            result = simulator.simulate_hit(shell, gun, target, optimal_angle)
            print(f"Hit: {'Yes' if result['hit'] else 'No'}")
            
            simulator.plot_trajectory(shell, gun, target, optimal_angle)
            
        elif choice == '3':
            # Batch simulation
            print("\nBatch simulation:")
            shell = simulator.shells["105mm HE"]
            gun = simulator.guns["M101 Howitzer"]
            target = Target(3000, 0, 8)
            
            simulator.batch_simulation(shell, gun, target, (15, 65), 2.0)
            
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
