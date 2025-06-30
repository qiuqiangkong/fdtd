import numpy as np
import random
import time


class FDTD2D:
    def __init__(
        self, 
        duration: float = 0.1, 
        skip: int = 10,
        verbose: bool = False
    ):
        r"""FDTD wave simulator."""

        self.duration = duration
        self.skip = skip
        self.verbose = verbose
        
        self.dx = 0.1  # Should be smaller than λ
        self.dy = 0.1
        self.nx = 100
        self.ny = 100
        
        self.c = 343.
        self.dt = self.dx / self.c / 3  # CFL condition
        self.nt = round(self.duration / self.dt)

    def simulate(self) -> dict:

        # Initialize
        u = np.zeros((self.nx, self.ny))
        u_prev = np.zeros_like(u)
        u_next = np.zeros_like(u)
        
        # Boundary and obstacles
        bound = self.sample_boundary()  # shape: (x, y)
        
        # Sample source
        u = self.sample_source()  # shape: (x, y)
        
        us = []

        # Simulate
        for t in range(self.nt):

            us.append(u.copy())
            
            # Calculate u_next by discretized 2D wave equation
            laplacian = (1. / self.dx**2) * (
                u[2:, 1:-1] + u[:-2, 1:-1] + 
                u[1:-1, 2:] + u[1:-1, :-2] - 
                4 * u[1:-1, 1:-1]
            )

            u_next[1:-1, 1:-1] = 2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (self.c**2 * self.dt**2) * laplacian
            
            # Apply rigid boundary
            u_next[bound] = 0
            u[bound] = 0
            u_prev[bound] = 0

            # Update state
            u_prev[:] = u
            u[:] = u_next
            
            if self.verbose:
                print("{}/{}".format(t, self.nt))

        us = np.array(us[0 :: self.skip])  # shape: (t, x ,y)
        ts = np.arange(self.nt)[0 :: self.skip] * self.dt
        
        data = {
            "boundary": bound.astype(np.float32),
            "t": ts.astype(np.float32),
            "u": us.astype(np.float32),
        }

        return data

    def sample_boundary(self) -> np.ndarray:
        r"""Sample boundary"""

        bound = np.zeros((self.nx, self.ny), dtype=bool)

        # Boundary
        bound[0 : 2, :] = 1
        bound[-2:, :] = 1
        bound[:, 0 : 2] = 1
        bound[:, -2:] = 1

        # Obstacles
        cx = round(0.25 * self.nx)  # center of obstacle
        cy = round(0.5 * self.ny)
        wx = random.randint(round(0.1 * self.nx), round(0.4 * self.nx))  # width of obstacle
        wy = random.randint(round(0.2 * self.ny), round(0.8 * self.ny))
        
        bound[cx - wx // 2 : cx + wx // 2, cy - wy // 2 : cy + wy // 2] = 1

        return bound

    def sample_source(self) -> np.ndarray:
        r"""Sample source."""

        cx = random.randint(60, 90)
        cy = random.randint(10, 90)
        x = np.arange(self.nx)[:, None]
        y = np.arange(self.ny)[None, :]
        sigma = 1.
        u = np.exp(-((x - cx)**2 + (y - cy)**2) / (2*sigma**2))

        return u


def visualize(x: np.ndarray, bound: np.ndarray) -> None:
    
    fig, ax = plt.subplots()
    cax = ax.imshow(x[0], cmap='jet', vmin=-1, vmax=1)
    fig.colorbar(cax)

    def update(frame):
        cax.set_data(frame)
        return cax,

    # from IPython import embed; embed(using=False); os._exit(0)
    frames = np.clip(x + bound, -1, 1)

    ani = animation.FuncAnimation(fig, func=update, frames=frames, interval=20)
    writer = animation.FFMpegWriter(fps=24, bitrate=5000)

    out_path = "out.mp4"
    ani.save(out_path, writer=writer)
    print("Write sound filed MP4 to {}".format(out_path))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # FDTD simulator
    sim = FDTD2D(duration=0.1, verbose=True)

    # Simulate
    t = time.time()
    data = sim.simulate()
    simulation_time = time.time() - t

    # Print
    u = data["u"]  # (t, x, y)
    bound = data["boundary"]  # (x, y)
    print("Sound field shape: {}".format(u.shape))
    print("Simulation time: {:.2f} s".format(simulation_time))

    # Write video for visualization
    visualize(u, bound)