import numpy as np
import sys
sys.path.insert(0, '../environment')
sys.path.insert(0, '../controller')
from Quadcopter_Dynamics import environment
from adaptive_controller import QuadcopterController


print("="*60)
print("TESTING ENVIRONMENT AND CONTROLLER")
print("="*60)

print("\n1. Creating environment...")
try:
    env = environment(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)
    print("   ✓ Environment created successfully")
    print(f"   Mass: {env.m} kg")
    print(f"   Gravity: {env.g} m/s²")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

print("\n2. Creating controller...")
try:
    ctrl = QuadcopterController(mass=1.2, g=9.81)
    print("   ✓ Controller created successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

print("\n3. Testing controller output...")
try:
    state = np.zeros(12)
    state[2] = 0.1  # Start at 0.1m altitude
    target = np.array([0, 0, 3])  # Target: hover at 3m

    u = ctrl.controller(state, target)
    print(f"   ✓ Controller computed control")
    print(f"   Control: F={u[0]:.2f}N, τφ={u[1]:.4f}, τθ={u[2]:.4f}, τψ={u[3]:.4f}")

    # Check reasonable values
    if u[0] < 0 or u[0] > 50:
        print(f"   ⚠ Warning: Thrust seems unreasonable")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

print("\n4. Testing environment step...")
try:
    state_dot = env.step(state, u)
    print(f"   ✓ Environment stepped successfully")
    print(f"   State derivative shape: {state_dot.shape}")
    print(f"   Vertical acceleration: {state_dot[5]:.4f} m/s²")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n5. Running 10-step simulation loop...")
try:
    dt = 0.01
    for i in range(10):
        target = np.array([0, 0, 3])
        u = ctrl.controller(state, target)
        state_dot = env.step(state, u)
        state = state + state_dot * dt

        if i == 0 or i == 9:
            print(f"   Step {i}: z={state[2]:.4f}m, vz={state[5]:.4f}m/s")

    print("   ✓ 10-step loop completed successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n6. Running 100-step simulation loop...")
try:
    import time
    start = time.time()

    state = np.zeros(12)
    state[2] = 0.1

    for i in range(100):
        if i % 10 == 0:
            print(f"   Iteration {i}/100", flush=True)
        if i >= 10 and i <= 12:
            print(f"     [DEBUG] i={i}, theta={np.degrees(state[7]):.1f}°, phi={np.degrees(state[6]):.1f}°", flush=True)
        target = np.array([2*i*dt, 0, 3])  # Moving target
        u = ctrl.controller(state, target)
        if i >= 10 and i <= 12:
            print(f"     [DEBUG] After controller, before step", flush=True)
        state_dot = env.step(state, u)
        if i >= 10 and i <= 12:
            print(f"     [DEBUG] After step, before update", flush=True)
        state = state + state_dot * dt

    elapsed = time.time() - start
    print(f"   ✓ 100 steps completed in {elapsed:.3f}s")
    print(f"   Average: {elapsed/100*1000:.2f}ms per step")
    print(f"   Estimated for 1000 steps: {elapsed*10:.1f}s")
    print(f"   Final position: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}) m")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
