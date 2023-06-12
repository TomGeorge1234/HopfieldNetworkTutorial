import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import copy

plt.rcParams["animation.html"] = "jshtml"


class BaseHopfieldNetwork:
    """Base class for our Hopfield Network (Modern) Hopfield Network"""

    def __init__(self, patterns_dict):
        """Initialises the Hopfield network with a set of patterns. It loads
        Args:
            • patterns: a dictionary containing the patterns to be stored in the network labelled by their names. Patterns can be any shape, they will be flattened into vectors during initialisation.
        """

        # Convert the patterns dictionary into a an array of flattened patterns
        self.patterns_dict = patterns_dict
        self.pattern_names = list(patterns_dict.keys())
        self.patterns = np.array(list((patterns_dict.values())))
        self.pattern_shape = patterns_dict[self.pattern_names[0]].shape
        # Some useful variables
        self.N_neurons = self.patterns[0].size
        self.N_patterns = self.patterns_dict.__len__()
        # Flatten the patterns into a matrix of shape (N_patterns, N_neurons)
        self.flattened_patterns = np.reshape(
            self.patterns, (self.N_patterns, self.N_neurons)
        )

        # Initialises a history dictionary
        self.history = {"state": [], "similarities": [], "energy": []}

        # Initialise the weights and state of the network
        # =================== YOUR CODE HERE ==============================
        self.w = NotImplemented  # <-- YOU WILL NEED TO WRITE THIS YOURSELF
        # =================================================================
        # =================== SOLUTION ==============================
        self.w = self.flattened_patterns.T @ self.flattened_patterns
        # ===========================================================

        self.set_state(random=True)  # initialises the state of the network
        return

    # =================== INITIALISE AND UPDATE NETWORK STATE  ======================
    def set_state(self, state=None, random=False):
        """Sets the state of the Hopfield network. If random = True, sets state to a random vector"""
        if random:
            self.state = np.random.choice([-1, 1], size=(self.N_neurons,))
        else:
            self.state = state.reshape(-1)
        self.save_history()

    def update_state(self):
        """Updates the state of the Hopfield network"""
        # =================== THIS HAS NOT BEEN WRITTEN< YOU WILL NEED TO WRITE THIS YOURSELF ==============================
        raise NotImplementedError

    # =================== ANALYSIS AND HISTORY FUNCTIONS ======================
    def save_history(self):
        """Calculates energy and similiarites then saves everything to the history of the Hopfield network"""
        self.similarities = self.get_similarities()
        self.energy = self.get_energy()

        self.history["state"].append(copy.deepcopy(self.state))
        self.history["similarities"].append(copy.deepcopy(self.similarities))
        self.history["energy"].append(copy.deepcopy(self.energy))

    def get_similarities(self, state=None):
        """Compares the state (defaults to the current state of the network to all stored patterns and returns a measure of similary between the current state and each stored pattern.
        This measure is taken as cos(theta) where theta is the angle between the current state vector and the stored pattern vectorin N-D space.
        """
        state = self.state if state is None else state
        return np.dot(self.flattened_patterns, self.state) / (
            np.linalg.norm(self.flattened_patterns, axis=1) * np.linalg.norm(self.state)
        )

    def get_energy(self, state=None):
        """Returns the energy of the network at a given state"""
        state = self.state if state is None else state
        return -0.5 * state @ self.w @ state

    # =================== PLOTTING FUNCTIONS ==============================
    def visualise(self, steps_back=0, fig=None, ax=None, title=None):
        """Visualises the state of the Hopfield network n_steps back (defaults to steps_back=0, i.e. current state)"""
        fig, ax = visualise_hopfield_network(
            self, steps_back=steps_back, fig=fig, ax=ax, title=title
        )
        return fig, ax

    def plot_energy(self, n_steps=None):
        """Plots the energy of the Hopfield network over time. n_steps=None defaults to _all_ steps"""
        fig, ax = plot_energy(self, n_steps=n_steps)
        return fig, ax

    def animate(self, n_steps=10, fps=10):
        """Animates the last n_steps of the Hopfield network. fps gives frames per socond of resulting animation"""
        anim = animate_hopfield_network(self, n_steps=n_steps, fps=fps)
        return anim


def plot_patterns(patterns: dict):
    """A function to plot patterns stored in a dictionary, on a grid.
    Args:
        • patterns (dict): a dictionary of patterns
    Returns:
        • fig, axs (tuple): a tuple of the figure and axes objects
    """
    # just calculate the grid shape
    N = patterns.__len__()  # number of patterns
    grid_size = max(i for i in range(1, int(np.sqrt(N) + 1)) if N % i == 0)
    grid_shape = [grid_size, int(N / grid_size)]
    grid_shape.sort()
    grid_shape = tuple(grid_shape)  # calculates the grid shape
    # plots the patterns onto the grid
    fig, axs = plt.subplots(
        grid_shape[0], grid_shape[1], figsize=(2 * grid_shape[1], 2 * grid_shape[0])
    )
    axs = axs.reshape(tuple(grid_shape))
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            name = list(patterns.keys())[i * grid_shape[1] + j]
            pattern = patterns[name]
            im = axs[i, j].imshow(pattern, cmap="Greys_r")
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)
            axs[i, j].set_title(name)
    # makes a colorbar
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
    return fig, axs


def visualise_hopfield_network(
    HopfieldNetwork, steps_back=0, fig=None, ax=None, title=None
):
    """Displays the state of the Hopfield network and a bar chart of similarites to all the stored patterns.
    Args:
        • HopfieldNetwork (HopfieldNetwork): the Hopfield network class storing the data to visualise
        • steps_back (int, optional): the number of steps back in the history to visualise. Defaults to 0 i.e. current state.
        • fig, ax: option fig and ax objects to plot onto. If None, new ones are created.
        • title (str, optional): the title of the figure. Defaults to None.
    """

    """Get the data to plot"""
    state = HopfieldNetwork.history["state"][-steps_back - 1]
    similarities = HopfieldNetwork.history["similarities"][-steps_back - 1]
    pattern_names = HopfieldNetwork.pattern_names
    pattern_shape = HopfieldNetwork.pattern_shape
    N_patterns = HopfieldNetwork.N_patterns

    """Create figure"""
    if ax is None:
        fig = plt.figure(figsize=(16, 4))
        ax0 = fig.add_axes([0, 0.02, 1 / 2, 0.88])
        ax1 = fig.add_axes([1 / 2, 1 / 3, 0.95 / 2, 0.9 - 1 / 3])
        ax = np.array([ax0, ax1])

    """Displays the state of the Hopfield network"""
    im = ax[0].imshow(state.reshape(pattern_shape), cmap="Greys_r")

    ax[0].set_title(title or "Network activity pattern", fontweight="bold")
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    """Displays the similarity of the Hopfield network state  """
    ax[1].set_title("Similarity to stored patterns")
    bars = ax[1].bar(np.arange(N_patterns), similarities)
    best_pattern_id = np.argmax(np.abs(similarities))
    best_pattern = HopfieldNetwork.patterns_dict[
        HopfieldNetwork.pattern_names[best_pattern_id]
    ]
    ax[1].set_xticks(np.arange(N_patterns))
    ax[1].set_xticklabels(pattern_names)
    ax[1].tick_params(axis="x", labelrotation=60)
    ax[1].axhline(0, color="black", lw=1)
    colors = ["C0"] * N_patterns
    colors[best_pattern_id] = "g"
    ax[1].tick_params(axis="x", which="major", pad=2)
    ax[1].set_ylim(top=1, bottom=-1)
    for i, (bar, tick) in enumerate(zip(bars, ax[1].get_xticklabels())):
        bar.set_facecolor(colors[i])
        tick.set_ha("right")
        if i == best_pattern_id:
            tick.set_color("g")

    """Display a small inset of the "best pattern" in the similarity bar chart"""
    width = HopfieldNetwork.N_patterns / 10
    inset_ax = ax[1].inset_axes(
        [best_pattern_id + 0.5, 0.5 - (1 - similarities[best_pattern_id]), width, 0.4],
        transform=ax[1].transData,
    )
    best_pattern = HopfieldNetwork.patterns_dict[
        HopfieldNetwork.pattern_names[best_pattern_id]
    ]
    inset_ax.imshow(best_pattern, cmap="Greys_r")
    plt.setp(inset_ax, xticks=[], yticks=[])
    for spine in list(inset_ax.spines.keys()):
        inset_ax.spines[spine].set_color("g")
        inset_ax.spines[spine].set_linewidth(2)

    return fig, ax


def plot_energy(HopfieldNetwork, n_steps=None):
    """Plots the energy of the Hopfield network over time"""
    fig, ax = plt.subplots(figsize=(5, 5))
    print("n_steps:", n_steps)
    if n_steps is None:
        n_steps = len(HopfieldNetwork.history["energy"]) - 2
        print("n_steps:", n_steps)
    energy = HopfieldNetwork.history["energy"][-n_steps - 1 :]
    t = np.arange(-n_steps - 1, 0)
    ax.scatter(t, energy, c=t, cmap="viridis", alpha=0.5)
    ax.plot(t, energy, c="k", lw=0.5, linestyle="-", zorder=0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_ylim(bottom=min(energy) - 5, top=max(energy) + 5)
    ax.set_xticks([-n_steps - 1, 0])
    ax.set_xticklabels([f"-{n_steps}", "0"])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    y_range = np.ptp(ax.get_ylim())
    first_inset_ax = ax.inset_axes(
        [
            -n_steps - 1 + 0.05 * n_steps,
            energy[0] + 0.05 * y_range,
            0.2 * n_steps,
            0.2 * y_range,
        ],
        transform=ax.transData,
    )
    first_inset_ax.imshow(
        HopfieldNetwork.history["state"][-n_steps - 1].reshape(
            HopfieldNetwork.pattern_shape
        ),
        cmap="Greys_r",
    )
    plt.setp(first_inset_ax, xticks=[], yticks=[])
    for spine in list(first_inset_ax.spines.keys()):
        first_inset_ax.spines[spine].set_color(matplotlib.colormaps["viridis"](0))
        first_inset_ax.spines[spine].set_linewidth(2.5)

    count = np.argmin(np.abs(np.array(energy) - (max(energy) - np.ptp(energy) / 3)))
    if count == 0:
        count = int(n_steps / 3)
    second_inset_ax = ax.inset_axes(
        [
            -n_steps - 1 + count + 0.05 * n_steps,
            energy[count] + 0.05 * y_range,
            0.2 * n_steps,
            0.2 * y_range,
        ],
        transform=ax.transData,
    )
    second_inset_ax.imshow(
        HopfieldNetwork.history["state"][-n_steps - 1 + count].reshape(
            HopfieldNetwork.pattern_shape
        ),
        cmap="Greys_r",
    )
    plt.setp(second_inset_ax, xticks=[], yticks=[])
    for spine in list(second_inset_ax.spines.keys()):
        second_inset_ax.spines[spine].set_color(
            matplotlib.colormaps["viridis"](count / n_steps)
        )
        second_inset_ax.spines[spine].set_linewidth(2.5)

    count = np.argmin(np.abs(np.array(energy) - (max(energy) - 2 * np.ptp(energy) / 3)))
    if count == 0:
        count = int(2 * n_steps / 3)
    third_inset_ax = ax.inset_axes(
        [
            -n_steps - 1 + count + 0.05 * n_steps,
            energy[count] + 0.05 * y_range,
            0.2 * n_steps,
            0.2 * y_range,
        ],
        transform=ax.transData,
    )
    third_inset_ax.imshow(
        HopfieldNetwork.history["state"][-n_steps - 1 + count].reshape(
            HopfieldNetwork.pattern_shape
        ),
        cmap="Greys_r",
    )
    plt.setp(third_inset_ax, xticks=[], yticks=[])
    for spine in list(third_inset_ax.spines.keys()):
        third_inset_ax.spines[spine].set_color(
            matplotlib.colormaps["viridis"](count / n_steps)
        )
        third_inset_ax.spines[spine].set_linewidth(2.5)

    last_inset_ax = ax.inset_axes(
        [0 + 0.05 * n_steps, energy[-1] + 0.05 * y_range, 0.2 * n_steps, 0.2 * y_range],
        transform=ax.transData,
    )
    last_inset_ax.imshow(
        HopfieldNetwork.history["state"][-1].reshape(HopfieldNetwork.pattern_shape),
        cmap="Greys_r",
    )
    plt.setp(last_inset_ax, xticks=[], yticks=[])
    for spine in list(last_inset_ax.spines.keys()):
        last_inset_ax.spines[spine].set_color(matplotlib.colormaps["viridis"](0.9999))
        last_inset_ax.spines[spine].set_linewidth(2.5)
    return fig, ax


def animate_hopfield_network(HopfieldNetwork, n_steps=10, fps=10):
    """Makes an animation of the last n states (drawn from the history) of the Hopfield network
    Args:
        • HopfieldNetwork (HopfieldNetwork): the Hopfield network class storing the data to animate
        • n_steps (int, optional): _description_. Defaults to 10.
    """
    fig, ax = HopfieldNetwork.visualise()

    def animate(i, fig, ax):
        """The function that is called at each step of the animation.
        This just clears the axes and revisualises teh state of the network at the next step.
        """
        ax[0].clear()
        ax[1].clear()
        steps_back = n_steps - i
        fig, ax = HopfieldNetwork.visualise(steps_back=steps_back, fig=fig, ax=ax)
        fig.suptitle("Step %d" % i)
        # fig.set_tight_layout(True)
        plt.close()

    anim = matplotlib.animation.FuncAnimation(
        fig, animate, fargs=(fig, ax), frames=n_steps, interval=1000 / fps, blit=False
    )
    return anim


def mask_pattern(pattern):
    """Masks all but the top left hand corner of a pattern"""
    mask = np.zeros_like(pattern)
    mask[: pattern.shape[0] // 2, : pattern.shape[1] // 2] = 1
    return pattern * mask


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def log_sum_exp(x, beta=1):
    """Computes the log of the sum of the exponential of the input. We only use this for modern hopfield networks right at the end"""
    return np.log(np.sum(np.exp(beta * x))) / beta
