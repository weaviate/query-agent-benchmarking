"""Mock repository implementations for testing."""


class MockResultRepository:
    """Mock result repository that stores results in memory."""

    def __init__(self):
        self.saved_trials = []
        self.saved_ask_trials = []
        self.saved_metrics = []
        self.saved_aggregated = []

    def save_trial_results(self, results, config, trial_number):
        self.saved_trials.append((results, config, trial_number))

    def save_ask_trial_results(self, results, config, trial_number, alignment_scores=None):
        self.saved_ask_trials.append((results, config, trial_number, alignment_scores))

    def save_trial_metrics(self, metrics, config, trial_number):
        self.saved_metrics.append((metrics, config, trial_number))

    def save_aggregated_results(self, aggregated_metrics, config):
        self.saved_aggregated.append((aggregated_metrics, config))
