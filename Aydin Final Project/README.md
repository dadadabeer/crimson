# Projectile Motion Simulator

A comprehensive Python simulation for calculating optimal firing angles to hit targets using different shells and guns. The simulator incorporates realistic physics including gravity, air resistance, and projectile characteristics.

## Features

- **Multiple Shell Types**: HE (High Explosive), APFSDS (Armor-Piercing Fin-Stabilized Discarding Sabot), Mortar rounds
- **Different Gun Systems**: Howitzers, Tank guns, Mortars with realistic elevation limits
- **Physics Simulation**: Includes gravity, air resistance, and drag calculations
- **Target Hit Detection**: Determines if shots hit targets based on size and position
- **Optimal Angle Calculation**: Automatically finds the best firing angle for any target
- **Trajectory Visualization**: Plots projectile paths with matplotlib
- **Interactive Mode**: User-friendly interface for experimentation
- **Batch Simulation**: Test multiple angles to find the best solution

## Physics Model

The simulation uses a realistic physics model that accounts for:

- **Gravity**: 9.81 m/s² downward acceleration
- **Air Resistance**: Drag force proportional to velocity squared
- **Shell Properties**: Mass, caliber, muzzle velocity, drag coefficient
- **Gun Characteristics**: Elevation limits, barrel length
- **Target Parameters**: Distance, elevation, size

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

```bash
python simulation.py
```

### Main Menu Options

1. **Interactive Simulation**: Choose shells, guns, and targets interactively
2. **Quick Demo**: See a pre-configured example
3. **Batch Simulation**: Test multiple angles systematically
4. **Exit**: Close the program

### Interactive Mode

1. Select a shell type (mass, velocity, drag characteristics)
2. Choose a gun system (elevation limits)
3. Set target parameters (distance, elevation, size)
4. The simulator calculates the optimal firing angle
5. View results and trajectory plots

### Shell Types Available

- **105mm HE**: 15kg, 650 m/s, good for medium range
- **155mm HE**: 43.5kg, 827 m/s, long range artillery
- **120mm APFSDS**: 8.1kg, 1700 m/s, high velocity tank round (M1A2 Abrams)
- **81mm Mortar**: 4.2kg, 300 m/s, high angle indirect fire

### Gun Systems Available

- **M101 Howitzer**: 65° max elevation, good for indirect fire
- **M198 Howitzer**: 72° max elevation, modern artillery
- **M1A2 Tank**: 20° max elevation, direct fire
- **81mm Mortar**: 85° max elevation, high angle fire

## Example Scenarios

### Scenario 1: Long Range Artillery
- Shell: 155mm HE
- Gun: M198 Howitzer
- Target: 8000m away, ground level
- Result: Optimal angle ~45°, time of flight ~45 seconds

### Scenario 2: Medium Range Direct Fire
- Shell: 105mm HE
- Gun: M101 Howitzer
- Target: 3000m away, ground level
- Result: Optimal angle ~25°, time of flight ~8 seconds

### Scenario 3: High Angle Mortar
- Shell: 81mm Mortar
- Gun: 81mm Mortar
- Target: 1500m away, elevated position
- Result: Optimal angle ~75°, time of flight ~25 seconds

### Scenario 4: Tank Direct Fire
- Shell: 120mm APFSDS
- Gun: M1A2 Tank
- Target: 1000m away, ground level
- Result: Optimal angle ~5°, time of flight ~1.5 seconds

## Physics Equations

The simulation solves the following differential equations:

```
dx/dt = vx
dy/dt = vy
dvx/dt = -0.5 * ρ * v² * Cd * A / m
dvy/dt = -g - 0.5 * ρ * v² * Cd * A / m
```

Where:
- ρ (rho) = air density
- v = velocity magnitude
- Cd = drag coefficient
- A = cross-sectional area
- m = mass
- g = gravitational acceleration

## Customization

You can easily add new shells and guns by modifying the `ProjectileSimulator` class:

```python
# Add new shell
self.shells["Custom Shell"] = Shell("Custom Shell", mass, caliber, velocity, drag_coef, area)

# Add new gun
self.guns["Custom Gun"] = Gun("Custom Gun", max_elev, min_elev, barrel_length)
```

## Output

The simulator provides:
- Optimal firing angle calculation
- Hit/miss determination
- Accuracy percentage
- Miss distance measurement
- Trajectory visualization
- Time of flight
- Impact coordinates

## Assumptions

- Flat terrain (no obstacles)
- Constant air density
- No wind effects (can be added)
- No Coriolis effect
- No temperature/pressure variations
- Target azimuth is known (north of firing position)

## Future Enhancements

- Wind effects
- Terrain elevation
- Multiple target scenarios
- Shell fragmentation patterns
- Weather conditions
- Moving targets
- 3D visualization

## License

This project is for educational purposes. Use responsibly and in accordance with applicable laws and regulations.
