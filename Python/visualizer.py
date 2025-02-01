import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # For systems without display

class SortVisualizer:
    def __init__(self, data, title="Sorting Visualization"):
        self.data = data
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.bars = self.ax.bar(range(len(data)), data, color='blue')
        self.ax.set_title(title)
        self.frames = []
        
    def update(self, frame):
        """Update function for animation"""
        for rect, val in zip(self.bars, frame):
            rect.set_height(val)
        return self.bars
        
    def add_frame(self, data):
        """Add a frame to the animation"""
        self.frames.append(data.copy())
        
    def save_animation(self, filename, fps=30):
        """Save the animation as a video file"""
        anim = FuncAnimation(
            self.fig, 
            self.update,
            frames=self.frames,
            interval=1000/fps,
            blit=True
        )
        anim.save(filename, writer='pillow', fps=fps)
        plt.close()

class LiveVisualizer:
    def __init__(self, size):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.size = size
        plt.ion()  # Enable interactive mode
        
    def update(self, data, title):
        """Update the visualization in real-time"""
        self.ax.clear()
        self.ax.bar(range(len(data)), data)
        self.ax.set_title(title)
        plt.pause(0.01)