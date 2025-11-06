# projectile_demo_safe.py
# Neutral projectile-motion demo (ball/cannonball), no target-aim solver.
# - 2D plane, flat ground, optional drag
# - RK4 integrator (accurate) or Semi-Implicit Euler (fast)
# - Interpolates ground impact for precise range
#
# pip install numpy matplotlib

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable

GRAVITY = 9.81  # m/s^2
RHO_AIR = 1.225 # kg/m^3 (sea level)

@dataclass
class Ball:
    mass: float = 0.15                 # kg (e.g., baseball-ish)
    drag_coefficient: float = 0.3      # dimensionless
    cross_section_area: float = 0.0042 # m^2 (~7.3 cm diameter)

@dataclass
class Launcher:
    v0: float = 30.0       # m/s
    min_angle: float = 0.0 # deg
    max_angle: float = 85.0# deg

@dataclass
class SimConfig:
    dt: float = 0.005          # s
    use_rk4: bool = True       # RK4 vs semi-implicit Euler
    with_drag: bool = True     # include drag
    max_time: float = 120.0    # s safety cap

def drag_accel(vx: float, vy: float, ball: Ball) -> Tuple[float,float]:
    """Drag acceleration components (opposite to velocity)."""
    v = np.hypot(vx, vy)
    if v == 0:
        return 0.0, 0.0
    k = 0.5 * RHO_AIR * ball.drag_coefficient * ball.cross_section_area / ball.mass
    ax = -k * v * vx
    ay = -k * v * vy
    return ax, ay

def dynamics(state: np.ndarray, ball: Ball, cfg: SimConfig) -> np.ndarray:
    """
    state = [x, y, vx, vy]
    returns derivative dstate/dt
    """
    x, y, vx, vy = state
    if cfg.with_drag:
        ax_d, ay_d = drag_accel(vx, vy, ball)
    else:
        ax_d, ay_d = 0.0, 0.0
    ax = ax_d
    ay = -GRAVITY + ay_d
    return np.array([vx, vy, ax, ay], dtype=float)

def rk4_step(state: np.ndarray, h: float, deriv: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    k1 = deriv(state)
    k2 = deriv(state + 0.5*h*k1)
    k3 = deriv(state + 0.5*h*k2)
    k4 = deriv(state + h*k3)
    return state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def semi_implicit_euler_step(state: np.ndarray, h: float, deriv: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    # update velocity using acceleration at current state, then position with new velocity
    x, y, vx, vy = state
    _, _, ax, ay = deriv(state)
    vx_new = vx + h*ax
    vy_new = vy + h*ay
    x_new  = x + h*vx_new
    y_new  = y + h*vy_new
    return np.array([x_new, y_new, vx_new, vy_new], dtype=float)

def integrate_trajectory(ball: Ball, launcher: Launcher, angle_deg: float, cfg: SimConfig
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns x(t), y(t), t, and precise impact_x via interpolation.
    """
    theta = np.deg2rad(angle_deg)
    vx0 = launcher.v0 * np.cos(theta)
    vy0 = launcher.v0 * np.sin(theta)

    state = np.array([0.0, 0.0, vx0, vy0], dtype=float)
    xs, ys, ts = [0.0], [0.0], [0.0]

    def D(s): return dynamics(s, ball, cfg)

    step = rk4_step if cfg.use_rk4 else semi_implicit_euler_step
    t = 0.0
    impact_x = None

    while t < cfg.max_time:
        s_prev = state.copy()
        t_prev = t
        state  = step(state, cfg.dt, D)
        t     += cfg.dt

        xs.append(state[0])
        ys.append(state[1])
        ts.append(t)

        # ground-crossing interpolation (first time y <= 0 after being > 0)
        if ys[-2] > 0.0 and ys[-1] <= 0.0:
            x0, y0, t0 = xs[-2], ys[-2], ts[-2]
            x1, y1, t1 = xs[-1], ys[-1], ts[-1]
            # linear interpolation on y
            frac = y0 / (y0 - y1)  # in (0,1]
            impact_x = x0 + frac*(x1 - x0)
            # also refine the last sample to land exactly on ground for pretty plots
            xs[-1] = impact_x
            ys[-1] = 0.0
            ts[-1] = t0 + frac*(t1 - t0)
            break

        # if the arc is going up forever (shouldn't), safety exit
        if len(ys) > 2 and ys[-1] > ys[-2] and ys[-2] > ys[-3] and ys[-1] > 1e6:
            break

    if impact_x is None:
        # if it never crossed back to ground (e.g., dt too large), fall back to last x
        impact_x = xs[-1]
    return np.array(xs), np.array(ys), np.array(ts), impact_x

def plot_trajectories(ball: Ball, launcher: Launcher, angles: List[float], cfg: SimConfig):
    plt.figure(figsize=(10,6))
    max_x = 0.0
    max_y = 0.0
    for a in angles:
        x, y, t, impact_x = integrate_trajectory(ball, launcher, a, cfg)
        plt.plot(x, y, label=f"{a:.1f}° (range {impact_x:.1f} m)")
        max_x = max(max_x, impact_x)
        max_y = max(max_y, np.max(y))
        # draw a faint vertical line at impact
        plt.axvline(impact_x, color='k', linestyle=':', alpha=0.2)

    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel("Horizontal distance (m)")
    plt.ylabel("Height (m)")
    plt.title(f"Projectile trajectories (v0={launcher.v0} m/s, drag={'on' if cfg.with_drag else 'off'}, integrator={'RK4' if cfg.use_rk4 else 'Semi-Implicit Euler'})")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.xlim(0, max_x*1.1 if max_x > 0 else 10)
    plt.ylim(0, max_y*1.1 if max_y > 0 else 10)
    plt.tight_layout()
    plt.show()

def main():
    ball = Ball()                  # tweak mass/area/Cd for different “balls”
    launcher = Launcher(v0=30.0)   # launch speed
    cfg = SimConfig(dt=0.005, use_rk4=True, with_drag=True)

    # Try a few angles
    angles = [20, 30, 40, 45, 55]
    for a in angles:
        x, y, t, impact_x = integrate_trajectory(ball, launcher, a, cfg)
        print(f"Angle {a:>5.1f}° → range = {impact_x:8.2f} m, peak = {np.max(y):7.2f} m, flight = {t[-1]:6.2f} s")

    plot_trajectories(ball, launcher, angles, cfg)

if __name__ == "__main__":
    main()
