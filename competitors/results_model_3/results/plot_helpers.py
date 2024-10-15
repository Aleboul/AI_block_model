# Header:
# This script contains various utility functions and plotting routines for statistical analysis.
# It includes functions for plotting trajectories, calculating statistical intervals, hazard rates 
# from run lengths, and fitting a truncated Laplace distribution. Additionally, there are helper 
# functions to configure plot styles, set marker properties, and toggle LaTeX support in Matplotlib.

# Import necessary libraries
import itertools  # For efficient looping and grouping
from collections import defaultdict  # For dictionary with default values
import matplotlib  # For general plotting
import matplotlib.pyplot as plt  # For specific plot creation
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import seaborn as sns  # For enhanced data visualization
from scipy import stats  # For statistical functions
from scipy.optimize import minimize_scalar  # For optimization

# Define color schemes for plots
colors = ['#EF476F', '#118AB2', '#06D6A0', '#073B4C', '#FFD166', "#FFACBB", "#33ACD4"]  # Main colors
lighter_colors = ["#FFACBB", "#33ACD4"]  # Lighter tones for secondary elements

# Function to initialize plot settings (set font, colors, etc.)
def init_plot():
    pd.set_option('display.max_columns', 50)  # Show up to 50 columns in DataFrames
    sns.set_context('talk')  # Set Seaborn context for larger font size
    sns.set_palette(colors)  # Set the color palette for Seaborn plots
    matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})  # Set font to Helvetica
    matplotlib.rcParams['pdf.fonttype'] = 42  # Set PDF font type
    matplotlib.rcParams['ps.fonttype'] = 42  # Set PostScript font type

# Function to set marker properties for plots
def marker_settings(edgecolor, markersize=10):
    return dict(
        color=edgecolor,  # Set marker color
        markerfacecolor='white',  # Set marker face color
        markeredgecolor=edgecolor,  # Set marker edge color
        markersize=markersize,  # Set marker size
        markeredgewidth=2  # Set marker edge width
    )

# Function to plot trajectories of simulations
def plot_trajectories(ax, trajs, label='Simulation', estimator=np.mean, linestyle='--', color=None, markersize=10):
    # If a color is provided, use customized marker settings
    if color:
        extra_settings = marker_settings(color)
    else:
        extra_settings = dict(marker='o', markersize=markersize)

    total_years = 20  # Set the total number of years for the simulation
    # Plot the simulation trajectories with lineplot
    sns.lineplot(
        x=itertools.chain(*[itertools.repeat(i, trajs.shape[1]) for i in range(total_years + 1)]),  # X-values
        y=trajs.flatten(),  # Flattened Y-values from the trajectory data
        estimator=estimator,  # Statistical estimator (mean by default)
        label=label,  # Label for the plot
        ax=ax,  # Axis on which to plot
        linestyle=linestyle,  # Line style
        lw=4,  # Line width
        **extra_settings  # Extra marker settings
    )

# Function to plot vertical cutoff lines on the plot
def plot_cutoffs(ax, cutoffs, color):
    for x in cutoffs:
        ax.axvline(x, color=color, linestyle='--')  # Plot vertical lines at cutoff points

# Function to format and print p-values
def print_pval(pval):
    if pval < 0.001:
        return 'p<0.001'
    if pval < 0.01:
        return f"p={pval:.03f}"
    return f"p={pval:.02f}"

# Function to calculate Wald 95% confidence intervals
def wald_interval_95(p, n):
    err = np.sqrt(p * (1 - p) / n)  # Standard error calculation
    return (p - 1.96 * err, p + 1.96 * err)  # Return confidence interval

# Function to count consecutive runs of zeros in a column
def count_zero_runs(column):
    return [len(list(v)) for k, v in itertools.groupby(column == 0) if k]

# Function to compute hazard rates based on run lengths of zeros
def hazard_from_runlengths(runlengths):
    zerohazard_counts = defaultdict(lambda: defaultdict(int))  # Dictionary to store counts
    zerohazard = np.zeros(max(runlengths))  # Array to store hazard values
    zerohazard_variances = np.zeros(max(runlengths))  # Array for variances

    # Populate counts of zero-run lengths
    for run in runlengths:
        for i in range(1, run):
            zerohazard_counts[i]['Y'] += 1
        zerohazard_counts[run]['N'] += 1

    # Compute hazard values and variances
    for observed, counts in zerohazard_counts.items():
        N = counts['Y'] + counts['N']
        phat = counts['Y'] / N
        zerohazard[observed - 1] = phat
        zerohazard_variances[observed - 1] = phat * (1 - phat) / N

    return zerohazard_counts, zerohazard, zerohazard_variances

# Function to enable LaTeX support in Matplotlib for better text rendering
def enable_matplotlib_latex():
    plt.rcParams.update({
        "text.usetex": True,  # Enable LaTeX rendering
        "text.latex.preamble": r"\usepackage{amsmath}",  # Include amsmath for advanced formatting
    })

# Function to disable LaTeX support in Matplotlib
def disable_matplotlib_latex():
    plt.rcParams.update({
        "text.usetex": False,  # Disable LaTeX rendering
    })

# Function to compute the negative log-likelihood of the truncated Laplace distribution
def trunclaplace_negloglike(alpha, xs, ms):
    xs = np.array(xs)
    ms = np.array(ms)
    a = np.abs(xs - ms).sum()  # Compute absolute deviations
    b = np.log(2 - np.exp(-ms/alpha)).sum()  # Log-sum term for the distribution
    return len(xs) * np.log(alpha) + a / alpha + b  # Return the negative log-likelihood

# Function to fit a truncated Laplace distribution to data
def fit_trunc_laplace(data, ms):
    result = minimize_scalar(trunclaplace_negloglike, args=(data, ms), bounds=[0, 500], method='Bounded')
    return result.x  # Return the estimated alpha

# Function to find the mode of a dataset
def find_mode(vals, bins=50):
    counts, bars = np.histogram(vals, bins=bins)  # Create a histogram
    i = np.argmax(counts)  # Find the index of the maximum count
    mode = (bars[i] + bars[i+1]) / 2  # Compute the mode as the midpoint of the bin
    return mode

# Function to plot the fitted Laplace distribution on a plot
def plot_laplace_fit(ax, data, label, color, lw=5, linestyle="-"):
    mode = find_mode(data)  # Find the mode of the data
    alpha = fit_trunc_laplace(data, mode)  # Fit the truncated Laplace distribution
    rv = stats.laplace(loc=mode, scale=alpha)  # Create a Laplace distribution object
    x = np.linspace(-25, 25, 100)  # X-values for the plot
    # Plot the probability density function (PDF) of the fitted distribution
    plotted = ax.plot(
        x, rv.pdf(x), lw=lw, label=f'\\begin{{align*}}{label}\\widehat{{\\alpha}}&={align_and_format(alpha)} \\\\ \\widehat{{\\mu}}&={align_and_format(mode)}\\end{{align*}}',
        color=color, linestyle=linestyle
    )
    ax.semilogy()  # Set the y-axis to a logarithmic scale
    return plotted, alpha, mode  # Return the plot handle, alpha, and mode

# Helper function to format values for LaTeX-style output
def align_and_format(x):
    return f"{x:.2f}"  # Return the value formatted to two decimal places
