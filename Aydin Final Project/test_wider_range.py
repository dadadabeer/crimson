#!/usr/bin/env python3

from simulation import ProjectileSimulator, Target

def test_wider_angle_range():
    sim = ProjectileSimulator()
    shell = sim.shells['120mm APFSDS']
    gun = sim.guns['M1A2 Tank']
    
    print(f"Testing wider range of angles to find 750m hit")
    print(f"Shell: {shell.name}")
    print(f"Gun: {gun.name}")
    print()
    
    # Test angles from 0.070° to 0.090° in 0.001° increments
    angles_to_test = []
    for i in range(0, 21):
        angle = 0.070 + (i * 0.001)
        angles_to_test.append(angle)
    
    best_angle = None
    best_range = 0
    best_miss = float('inf')
    
    print("Angle (°) | Range (m) | Miss (m)")
    print("-" * 35)
    
    for angle in angles_to_test:
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, angle)
        
        # Find impact point
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        
        miss_distance = abs(impact_x - 750)
        
        print(f"{angle:7.3f}° | {impact_x:8.3f}m | {miss_distance:6.3f}m")
        
        if miss_distance < best_miss:
            best_miss = miss_distance
            best_angle = angle
            best_range = impact_x
    
    print("-" * 35)
    print(f"Best angle: {best_angle:.5f}°")
    print(f"Best range: {best_range:.3f}m")
    print(f"Best miss: {best_miss:.3f}m")
    
    # Now let's test if there's a sweet spot between angles
    print(f"\nTesting intermediate angles around {best_angle:.5f}°...")
    
    if best_angle > 0.070:
        # Test angle just below the best
        lower_angle = best_angle - 0.001
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, lower_angle)
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        lower_miss = abs(impact_x - 750)
        print(f"Angle {lower_angle:.5f}°: Range {impact_x:.3f}m, Miss {lower_miss:.3f}m")
    
    if best_angle < 0.090:
        # Test angle just above the best
        upper_angle = best_angle + 0.001
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, upper_angle)
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        upper_miss = abs(impact_x - 750)
        print(f"Angle {upper_angle:.5f}°: Range {impact_x:.3f}m, Miss {upper_miss:.3f}m")
    
    # Let's also test what happens if we go to much higher angles
    print(f"\nTesting higher angles to see if we can get closer to 750m...")
    
    high_angles = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    for angle in high_angles:
        x_pos, y_pos, times = sim.calculate_trajectory(shell, gun, angle)
        
        # Find impact point
        impact_x = 0
        for i in range(1, len(y_pos)):
            if y_pos[i] <= 0:
                impact_x = x_pos[i]
                break
        
        miss_distance = abs(impact_x - 750)
        
        print(f"Angle {angle:5.2f}°: Range {impact_x:8.1f}m, Miss {miss_distance:6.1f}m")
        
        if miss_distance < best_miss:
            best_miss = miss_distance
            best_angle = angle
            best_range = impact_x
    
    print(f"\nFinal best angle: {best_angle:.5f}°")
    print(f"Final best range: {best_range:.3f}m")
    print(f"Final best miss: {best_miss:.3f}m")

if __name__ == "__main__":
    test_wider_angle_range()
