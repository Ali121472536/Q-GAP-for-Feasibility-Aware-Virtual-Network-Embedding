import os
import numpy as np
import matplotlib.pyplot as plt


class Plotter:

    # ==========================================
    # Smooth Curve
    # ==========================================
    @staticmethod
    def smooth_curve(data, window=50):

        if len(data) < window:
            return data

        smoothed = np.convolve(
            data,
            np.ones(window) / window,
            mode='valid'
        )

        return smoothed

    # ==========================================
    # Generic Plot Function
    # ==========================================
    @staticmethod
    def plot_metric(

            data,
            ylabel,
            title,
            filename,
            smooth=True

    ):

        os.makedirs(
            "results",
            exist_ok=True
        )

        plt.figure(figsize=(8, 5))

        if smooth:
            data = Plotter.smooth_curve(data)

        plt.plot(
            data,
            linewidth=2
        )

        plt.xlabel("Episode")

        plt.ylabel(ylabel)

        plt.title(title)

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            f"results/{filename}",
            dpi=300
        )

        plt.close()