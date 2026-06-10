class FrameBuffer:
    def __init__(self, max_size=30):
        self.max_size = max_size
        self.frames = []

    def add(self, frame):
        self.frames.append(frame)
        if len(self.frames) > self.max_size:
            self.frames.pop(0)

    def is_ready(self):
        return len(self.frames) == self.max_size

    def get_frames(self):
        return self.frames.copy()

    def clear(self):
        self.frames = []
