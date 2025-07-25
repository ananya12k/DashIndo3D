import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_time(ts):
    """Parse MM-SS-MMMMMM to seconds"""
    try:
        parts = ts.split("-")
        minutes = int(parts[1])
        seconds = int(parts[2])
        micros = int(parts[3])
        return minutes * 60 + seconds + micros / 1e6
    except:
        return None


def compute_trajectory(obd_path):
    df = pd.read_csv(obd_path)

    # Convert timestamp to seconds
    df["time_sec"] = df["timestamp"].apply(parse_time)
    df = df.dropna()
    df = df.sort_values("time_sec").reset_index(drop=True)

    # Initialize
    positions = [np.array([0.0, 0.0])]  # x, y
    heading = 0.0  # in radians

    for i in range(1, len(df)):
        t1 = df["time_sec"].iloc[i - 1]
        t2 = df["time_sec"].iloc[i]
        dt = t2 - t1

        if dt <= 0:  # skip duplicates or reverse
            positions.append(positions[-1])
            continue

        speed_kmph = df["speed"].iloc[i]
        speed_mps = speed_kmph / 3.6

        dx = speed_mps * dt * np.cos(heading)
        dy = speed_mps * dt * np.sin(heading)

        new_pos = positions[-1] + np.array([dx, dy])
        positions.append(new_pos)

    return np.array(positions), df


if __name__ == "__main__":
    obd_csv_path = "obd.csv"
    traj, df = compute_trajectory(obd_csv_path)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label="Computed Trajectory")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Trajectory from OBD Speed Integration")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("obd_trajectory.png", dpi=300)
    plt.show()
