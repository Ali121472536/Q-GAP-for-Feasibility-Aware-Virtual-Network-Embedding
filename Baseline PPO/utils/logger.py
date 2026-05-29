import csv
import os


class CSVLogger:

    def __init__(self, filename):

        self.filename = filename

        os.makedirs(
            "results",
            exist_ok=True
        )

        self.filepath = (
            f"results/{filename}"
        )

        with open(
                self.filepath,
                mode='w',
                newline=''
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "Episode",

                "AcceptanceRatio",

                "Revenue",

                "Cost",

                "R2C",

                "CPUUtilization",

                "BWUtilization"
            ])

    # ==========================================
    # Log Row
    # ==========================================
    def log(

            self,
            episode,
            acceptance_ratio,
            revenue,
            cost,
            r2c,
            cpu_util,
            bw_util

    ):

        with open(
                self.filepath,
                mode='a',
                newline=''
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                episode,

                acceptance_ratio,

                revenue,

                cost,

                r2c,

                cpu_util,

                bw_util
            ])