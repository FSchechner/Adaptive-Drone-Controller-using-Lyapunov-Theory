import numpy as np
from simulator import QuadcopterSimulator
from trajectories import step_trajectory, hover_trajectory, circle_trajectory
from visualization import plot_results, plot_detailed_response, compute_performance_metrics

def main():
    sim = QuadcopterSimulator()

    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    t_span = [0, 20.0]
    dt = 0.01

    print("Running simulation...")
    results = sim.simulate(step_trajectory, initial_state, t_span, dt)

    print("\nComputing performance metrics...")
    compute_performance_metrics(results)

    print("\nGenerating plots...")
    plot_results(results)
    plot_detailed_response(results)

    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
