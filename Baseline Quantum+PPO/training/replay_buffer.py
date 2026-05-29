class ReplayBuffer:

    def __init__(self):
        self.buffer = []

    def store(self, transition):
        self.buffer.append(transition)

    def clear(self):
        self.buffer = []