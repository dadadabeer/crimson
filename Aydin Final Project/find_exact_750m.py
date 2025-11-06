#!/usr/bin/env python3

from simulation import ProjectileSimulator, Target

def find_exact_750m_angle():
    sim = ProjectileSimulator()
    shell = sim.shells['120mm APFSDS']
    gun = sim.guns['M1A2 Tank']
    target = Target(750, 2.2, 10)
    
    print(f"Finding exact angle to hit 750m with 120mm APFSDS")
    print(f"Target: {target.distance}m")
    print()
    
    # First, let's find the rough range where we get close to 750m
    print("Testing angles to find range around 750m...")
    
    # Test angles from 0.075° to 0.085° in 0.001° increments
    angles_to_test = []
    for i in range(-5, 6):
        angle = 0.08 + (i * 0.001)
        angles_to_test.append(angle)
    
    best_angle = None
    best_range = 0
    best_miss = float('inf')
    
    for angle in angles_to_test:
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, angle)
        
        # Find impact point
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        
        miss_distance = abs(impact_x - 750)
        
        print(f"Angle {angle:.5f}°: Impact at {impact_x:.3f}m, Miss: {miss_distance:.3f}m")
        
        if miss_distance < best_miss:
            best_miss = miss_distance
            best_angle = angle
            best_range = impact_x
    
    print(f"\nBest angle found: {best_angle:.5f}°")
    print(f"Range at best angle: {best_range:.3f}m")
    print(f"Miss distance: {best_miss:.3f}m")
    
    # Now let's find the exact angle for 750m
    print(f"\nSearching for exact 750m hit...")
    
    # We know the angle is between 0.079° and 0.081° based on previous results
    # Let's test very fine increments in this range
    start_angle = 0.079
    end_angle = 0.081
    step_size = 0.00001  # 0.00001° = 0.00000017 radians (very precise!)
    
    exact_angle = None
    exact_range = 0
    exact_miss = float('inf')
    
    for angle in range(int(start_angle * 100000), int(end_angle * 100000) + 1):
        test_angle = angle / 100000.0
        
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, test_angle)
        
        # Find impact point
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        
        miss_distance = abs(impact_x - 750)
        
        if miss_distance < exact_miss:
            exact_miss = miss_distance
            exact_angle = test_angle
            exact_range = impact_x
            
            # If we're within 1cm, that's good enough
            if miss_distance < 0.01:
                print(f"Found angle within 1cm: {test_angle:.6f}°")
                break
    
    print(f"\nExact angle for closest to 750m: {exact_angle:.6f}°")
    print(f"Range at exact angle: {exact_range:.4f}m")
    print(f"Miss distance: {exact_miss:.4f}m")
    
    # Now let's test a few angles around this to confirm
    print(f"\nTesting angles around {exact_angle:.6f}° for verification...")
    
    verification_angles = []
    for i in range(-2, 3):
        angle = exact_angle + (i * 0.00001)
        verification_angles.append(angle)
    
    for angle in verification_angles:
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, angle)
        
        # Find impact point
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        
        miss_distance = abs(impact_x - 750)
        
        print(f"Angle {angle:.6f}°: Impact at {impact_x:.4f}m, Miss: {miss_distance:.4f}m")
    
    return exact_angle, exact_range, exact_miss

if __name__ == "__main__":
    find_exact_750m_angle()
