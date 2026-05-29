class Evaluator:

    @staticmethod
    def compute_acceptance_ratio(success, total):

        if total == 0:
            return 0

        return success / total