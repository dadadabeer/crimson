#!/usr/bin/env python3

from simulation import ProjectileSimulator, Target

def test_exact_angle():
    sim = ProjectileSimulator()
    shell = sim.shells['120mm APFSDS']
    gun = sim.guns['M1A2 Tank']
    target = Target(750, 2.2, 10)
    
    print(f"Testing very fine angle increments around 0.08°")
    print(f"Target: {target.distance}m")
    print()
    
    # Test very fine angles around 0.08°
    base_angle = 0.08
    angles_to_test = []
    
    # Test angles from 0.075° to 0.085° in 0.001° increments
    for i in range(-5, 6):
        angle = base_angle + (i * 0.001)
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
        
        miss_distance = abs(impact_x - target.distance)
        
        print(f"Angle {angle:.5f}°: Impact at {impact_x:.3f}m, Miss: {miss_distance:.3f}m")
        
        if miss_distance < best_miss:
            best_miss = miss_distance
            best_angle = angle
            best_range = impact_x
    
    print(f"\nBest angle found: {best_angle:.5f}°")
    print(f"Range at best angle: {best_range:.3f}m")
    print(f"Miss distance: {best_miss:.3f}m")
    
    # Now test even finer increments around the best angle
    print(f"\nTesting ultra-fine increments around {best_angle:.5f}°...")
    
    ultra_fine_angles = []
    for i in range(-10, 11):
        angle = best_angle + (i * 0.0001)  # 0.0001° increments
        ultra_fine_angles.append(angle)
    
    ultra_best_angle = best_angle
    ultra_best_miss = best_miss
    
    for angle in ultra_fine_angles:
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, angle)
        
        # Find impact point
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        
        miss_distance = abs(impact_x - target.distance)
        
        if miss_distance < ultra_best_miss:
            ultra_best_miss = miss_distance
            ultra_best_angle = angle
    
    print(f"Ultra-fine best angle: {ultra_best_angle:.6f}°")
    print(f"Ultra-fine miss distance: {ultra_best_miss:.4f}m")

if __name__ == "__main__":
    test_exact_angle()
