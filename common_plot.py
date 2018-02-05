import re
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.spatial as spatial


def get_test_accuracy(log, top_k):
    iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', log)
    accuracy = re.findall(r'Test net output #\d: accuracy/top-{top_k} = (\d*.\d*)'.format(top_k=top_k), log)
    if len(accuracy)==0:
        accuracy = re.findall(r'Test net output #\d: top-{top_k} = (\d*.\d*)'.format(top_k=top_k), log)
    if len(accuracy)==0:
        accuracy = re.findall(r'Test net output #\d: loss/top-{top_k} = (\d*.\d*)'.format(top_k=top_k), log)
    if len(accuracy)==0:
        accuracy = re.findall(r'Test net output #\d: accuracy/top{top_k} = (\d*.\d*)'.format(top_k=top_k), log)
    if len(accuracy)==0:
        accuracy = re.findall(r'Test net output #\d: accuracy = (\d*.\d*)', log)
    iteration = [int(i) for i in iteration]
    accuracy = [float(i) for i in accuracy]
    return iteration, accuracy


def get_test_loss(log):
    iteration = re.findall(r'Iteration (\d*), Testing net ', log)
    loss = re.findall(r'Test net output #\d: loss = (\d*.\d*)', log)
    if len(loss)==0:
        loss = re.findall(r'Test net output #\d: loss/loss = (\d*.\d*)', log)
    iteration = [int(i) for i in iteration]
    loss = [float(i) for i in loss]
    return iteration, loss

def get_train_loss(log):
    iteration = re.findall(r'Iteration (\d*), lr = ', log)
    loss = re.findall(r'Train net output #\d: loss = (\d*.\d*)', log)
    iteration = [int(i) for i in iteration]
    loss = [float(i) for i in loss]
    return iteration, loss


def get_net_name(log):
    return re.findall(r"Solving (.*)\n", log)[0]


def parse_files(files, top_k=1, separate=False):
    data = {}
    for file in files:
        with open(file, 'r') as fp:
            log = fp.read()
            net_name = os.path.basename(file) if separate else get_net_name(log)
            if net_name not in data.keys():
                data[net_name] = {}
                data[net_name]["accuracy"] = {}
                data[net_name]["accuracy"]["accuracy"] = []
                data[net_name]["accuracy"]["iteration"] = []
                data[net_name]["loss"] = {}
                data[net_name]["loss"]["loss"] = []
                data[net_name]["loss"]["iteration"] = []
                data[net_name]["train_loss"] = {}
                data[net_name]["train_loss"]["loss"] = []
                data[net_name]["train_loss"]["iteration"] = []

            iteration, accuracy = get_test_accuracy(log, top_k)
            data[net_name]["accuracy"]["iteration"].extend(iteration)
            data[net_name]["accuracy"]["accuracy"].extend(accuracy)

            iteration, loss = get_test_loss(log)
            data[net_name]["loss"]["iteration"].extend(iteration)
            data[net_name]["loss"]["loss"].extend(loss)

            iteration, loss = get_train_loss(log)
            data[net_name]["train_loss"]["iteration"].extend(iteration)
            data[net_name]["train_loss"]["loss"].extend(loss)

    return data


def fmt(x, y):
    return 'x: {x:0.2f}\ny: {y:0.2f}'.format(x=x, y=y)


class FollowDotCursor(object):
    """Display the x,y location of the nearest data point.
    http://stackoverflow.com/a/4674445/190597 (Joe Kington)
    http://stackoverflow.com/a/20637433/190597 (unutbu)
    """
    def __init__(self, ax, x, y, formatter=fmt, offsets=(-20, 20)):
        try:
            x = np.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = np.asarray(mdates.date2num(x), dtype='float')
        y = np.asarray(y, dtype='float')
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        self.offsets = offsets
        y = y[np.abs(y - y.mean()) <= 3 * y.std()]
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='green', alpha=0.7)
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(x, y))
        self.dot.set_offsets((x, y))
        event.canvas.draw()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.75),
            arrowprops = dict(
                arrowstyle='->', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]


def plot_accuracy(top_k, data, value_at_hover=False):
    nets =  data.keys()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(nets))))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for net in nets:
        iteration = data[net]["accuracy"]["iteration"]
        accuracy = data[net]["accuracy"]["accuracy"]
        iteration, accuracy = (np.array(t) for t in zip(*sorted(zip(iteration, accuracy))))
        ax.plot(iteration, accuracy*100, color=next(colors), linestyle='-')
        if value_at_hover:
            cursor = FollowDotCursor(ax, iteration, accuracy*100)

    plt.legend(nets, loc='lower right')
    plt.title("Top {}".format(top_k))
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy [%]")
    plt.ylim(0,100)
    plt.grid()
    return plt


def plot_loss(data, value_at_hover=False):
    nets =  data.keys()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(nets))))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for net in nets:
        iteration = data[net]["loss"]["iteration"]
        loss = data[net]["loss"]["loss"]
        iteration, loss = (list(t) for t in zip(*sorted(zip(iteration, loss))))
        ax.scatter(iteration, loss, color=next(colors))
        if value_at_hover:
            cursor = FollowDotCursor(ax, iteration, loss)

    plt.legend(nets, loc='upper right')
    plt.title("Log Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.xlim(0)
    plt.grid()
    return plt

def plot_train_loss(data, value_at_hover=False):
    nets =  data.keys()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(nets))))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for net in nets:
        iteration = data[net]["train_loss"]["iteration"]
        loss = data[net]["train_loss"]["loss"]
        iteration, loss = (list(t) for t in zip(*sorted(zip(iteration, loss))))
        ax.scatter(iteration, loss, color=next(colors))
        if value_at_hover:
            cursor = FollowDotCursor(ax, iteration, loss)

    plt.legend(nets, loc='upper right')
    plt.title("Log Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.xlim(0)
    plt.grid()
    return plt

