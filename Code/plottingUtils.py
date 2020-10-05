import matplotlib.pyplot as plt
import numpy as np


def publishPlot(fig, save=False, dispPlot=True, fileName=None):
    if save:
        if fileName is None:
            raise ValueError("Provide a fileName when saving matrices")
        plt.savefig(fileName, dpi=300)

    if dispPlot:
        plt.show()
    else:
        plt.close(fig)


def makeTotalPerformancePlot(results, **kwargs):
    width = 0.5
    for key, val in kwargs.items():
        if key == 'width':
            width = val

    labels = results['exp']
    yVals = results['PVs found']
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, yVals, width)
    ax.set_ylabel('Number of PVs found')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('PVs found')

    fig.tight_layout()
    return fig


def makePerformanceByPVTypePlot(results, **kwargs):
    numExps = int((len(results.columns) - 2) / 2)
    width = 1 / (numExps + 2)
    labels = results['combination']
    x = np.arange(len(labels))

    figsize = (10, 5)
    for key, val in kwargs:
        if key == 'figsize':
            figsize = val

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(zorder=0)
    for expInd in range(numExps):
        l = str(results.columns[4 + 2*expInd])[23:]
        ax.bar(x + (-(numExps / 2) + expInd) * width, results.iloc[:, 4 + 2*expInd], width, label=l, zorder=3, edgecolor='k')

    ax.set_ylabel('Percentage PVs found')
    ax.set_title('Percentage PVs found by PV type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    return fig
