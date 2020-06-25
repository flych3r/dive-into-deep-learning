import matplotlib.pyplot as plt
from IPython import display


def show_images(imgs, num_rows, num_cols, titles=None, scale=2):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if 'asnumpy' in dir(img):
            img = img.asnumpy()
        if 'numpy' in dir(img):
            img = img.numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def semilogy(num_epochs, dists, legends, xlabel, ylabel, figsize):
    plt.figure(figsize=figsize)
    plt.semilogy(range(1, num_epochs + 1), dists[0], label=legends[0])
    plt.semilogy(range(1, num_epochs + 1), dists[1], ':', label=legends[1])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, title=None,
                 xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=None, nrows=1, ncols=1, figsize=(15, 10)):
        """Incrementally plot multiple lines."""
        if legend is None:
            legend = []
        if title is None:
            title = ''
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda to capture arguments
        self.config_axes = lambda: self.set_axes(
            xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title):
        """Set the axes for matplotlib."""
        self.axes[0].set_xlabel(xlabel)
        self.axes[0].set_ylabel(ylabel)
        self.axes[0].set_xscale(xscale)
        self.axes[0].set_yscale(yscale)
        self.axes[0].set_xlim(xlim)
        self.axes[0].set_ylim(ylim)
        if legend:
            self.axes[0].legend(legend)
        if title:
            self.axes[0].set_title(title)
        self.axes[0].grid()

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not self.fmts:
            self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
