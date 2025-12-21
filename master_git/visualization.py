import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import animation


def visualize_episode(obstacles, starts, goals, trajectories, save_path):
    """
    obstacles: numpy array [H, W] with 0 = free, 1 = obstacle
    starts: list of (x, y)
    goals: list of (x, y)
    trajectories: list of list-of-(x,y), len = num_agents
    save_path: path to mp4
    """

    H, W = obstacles.shape
    k = 20  # pixel size per cell
    colors = [tuple(np.random.randint(0, 255, 3)) for _ in starts]

    # prepare figure
    fig = plt.figure(figsize=(W/2, H/2))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    frames = []
    T = max(len(path) for path in trajectories)

    for t in range(T):
        img = Image.new("RGB", (W*k, H*k), "white")
        draw = ImageDraw.Draw(img)

        # Draw map
        for i in range(H):
            for j in range(W):
                if obstacles[i, j] == 1:
                    draw.rectangle((j*k, i*k, j*k+k, i*k+k), fill=(70, 80, 80))

        # Draw goals
        for agent_id, (gx, gy) in enumerate(goals):
            draw.ellipse((gy*k+4, gx*k+4, gy*k+k-4, gx*k+k-4), fill=colors[agent_id])

        # Draw agents
        for agent_id, traj in enumerate(trajectories):
            pos = traj[min(t, len(traj)-1)]
            x, y = pos
            draw.rectangle((y*k+2, x*k+2, y*k+k-2, x*k+k-2), fill=colors[agent_id])

        frame = plt.imshow(img, animated=True)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=300, blit=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer="ffmpeg")
    plt.close(fig)
