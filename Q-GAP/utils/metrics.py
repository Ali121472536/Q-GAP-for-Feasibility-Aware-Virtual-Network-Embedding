import numpy as np


class MetricsTracker:

    def __init__(self):

        self.acceptance_ratios = []

        self.revenues = []

        self.costs = []

        self.r2c_values = []

        self.cpu_utilization = []

        self.bw_utilization = []

    # ==========================================
    # Update Metrics
    # ==========================================
    def update(

            self,
            acceptance_ratio,
            revenue,
            cost,
            cpu_util,
            bw_util

    ):

        self.acceptance_ratios.append(
            acceptance_ratio
        )

        self.revenues.append(
            revenue
        )

        self.costs.append(
            cost
        )

        r2c = revenue / max(cost, 1)

        self.r2c_values.append(r2c)

        self.cpu_utilization.append(
            cpu_util
        )

        self.bw_utilization.append(
            bw_util
        )

    # ==========================================
    # Average Metrics
    # ==========================================
    def summary(self):

        return {

            "Average Acceptance Ratio":

                np.mean(
                    self.acceptance_ratios
                ),

            "Average Revenue":

                np.mean(
                    self.revenues
                ),

            "Average R2C":

                np.mean(
                    self.r2c_values
                ),

            "Average CPU Utilization":

                np.mean(
                    self.cpu_utilization
                ),

            "Average BW Utilization":

                np.mean(
                    self.bw_utilization
                )
        }