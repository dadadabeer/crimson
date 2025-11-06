#!/usr/bin/env python3

from simulation import ProjectileSimulator, Target

def test_simple_trajectory():
    sim = ProjectileSimulator()
    shell = sim.shells['120mm APFSDS']
    gun = sim.guns['M1A2 Tank']
    
    print(f"Testing 120mm APFSDS shell:")
    print(f"Mass: {shell.mass}kg")
    print(f"Velocity: {shell.muzzle_velocity}m/s")
    print(f"Drag coefficient: {shell.drag_coefficient}")
    print(f"Cross-sectional area: {shell.cross_sectional_area:.6f}m²")
    print()
    
    # Test different angles
    angles_to_test = [0, 5, 10, 15, 20]
    
    for angle in angles_to_test:
        print(f"Testing angle: {angle}°")
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, angle)
        
        if len(x_pos) > 1:
            print(f"  Trajectory points: {len(x_pos)}")
            print(f"  Final x: {x_pos[-1]:.2f}m")
            print(f"  Final y: {y_pos[-1]:.2f}m")
            print(f"  Time: {times[-1]:.2f}s")
            
            # Find where it hits ground
            for i, y in enumerate(y_pos):
                if y <= 0:
                    print(f"  Impact at x = {x_pos[i]:.2f}m, t = {times[i]:.2f}s")
                    break
        else:
            print(f"  No trajectory calculated")
        print()

if __name__ == "__main__":
    test_simple_trajectory()
