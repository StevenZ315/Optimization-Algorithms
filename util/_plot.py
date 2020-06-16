import matplotlib.pyplot as plt
import matplotlib.animation as animation

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, data, func=None, title=None):
        self.frames = len(data['solution'])
        self.data = data['solution']
        self.pbest = data['solution_best']
        self.boundary = data['boundary']

        # Global optimum if function is provided.
        if func:
            self.global_opt =func.solution()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        self.title = title

        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.frames, interval=500,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        # Current generation points.
        x = [point[0] for point in self.data[0]]
        y = [point[1] for point in self.data[0]]
        self.scatter = self.ax.scatter(x, y, marker='.', c='blue', alpha=0.5)

        # Current optimal solution.
        self.center = self.ax.scatter(self.pbest[0][0], self.pbest[0][1], marker='^', c='red')

        # Global optimum solution. (Benchmark)
        x_opt = [point[0] for point in self.global_opt]
        y_opt = [point[1] for point in self.global_opt]
        self.opt = self.ax.scatter(x_opt, y_opt, marker='o', c='green')

        # Axis settings.
        self.ax.set_title(self.title)
        self.ax.set_xlim(self.boundary[0])
        self.ax.set_ylim(self.boundary[1])
        self.ax.set_xlabel("Iteration: 0")
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scatter,

    def update(self, i):
        """Update the scatter plot."""
        self.scatter.set_offsets(self.data[i])
        self.center.set_offsets(self.pbest[i])
        self.ax.set_xlabel("Iteration: %d" % (i + 1))

        return self.scatter,

    def save(self, path):
        self.ani.save(path)



if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()