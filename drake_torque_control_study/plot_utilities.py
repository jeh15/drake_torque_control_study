import matplotlib.pyplot as plt


def legend_for(xs):
    n = xs.shape[1]
    labels = [f"{i}" for i in range(1, n + 1)]
    plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))


def reset_color_cycle():
    # https://stackoverflow.com/a/39116381/7829525
    plt.gca().set_prop_cycle(None)
