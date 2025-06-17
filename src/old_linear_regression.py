import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from statsmodels.graphics.gofplots import ProbPlot

def plot_residuals(axes, res, yfit, n_samp=0):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    x: x values
    ytrue: y values
    yfit: fitted/predicted y values
    res[optional]: Residuals, used for resampling
    n_samp[optional]: number of resamples """
    
    # For every random resampling
    for i in range(n_samp):
        # 1. resample indices from Residuals
        samp_res_id = random.sample(list(res), len(res))
        
        # 2. Average of Residuals, smoothed using LOWESS
        sns.regplot(x=yfit, y=samp_res_id,
            scatter=False, ci=False, lowess=True,
            line_kws={'color': 'lightgrey', 'lw': 1, 'alpha': 0.8})
        # 3. Repeat again for n_samples
        
    dataframe = pd.concat([yfit, res], axis=1)
    axes = sns.residplot(x=yfit, y=res, data=dataframe,
    lowess=True, scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    axes.set_title('Turkey-Anscombe: Residuals vs Fitted')
    axes.set_ylabel('Residuals')
    axes.set_xlabel('Fitted Values')
    
def plot_QQ(axes, res_standard, n_samp=0):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    res_standard: standardized residuals
    n_samp[optional]: number of resamples """
    # QQ plot instance
    QQ = ProbPlot(res_standard)
    
    # Split the QQ instance in the seperate data
    qqx = pd.Series(sorted(QQ.theoretical_quantiles), name="x")
    qqy = pd.Series(QQ.sorted_data, name="y")
    
    if n_samp != 0:
        # Estimate the mean and standard deviation
        mu = np.mean(qqy)
        sigma = np.std(qqy)
        # For ever random resampling
        
        for lp in range(n_samp):
            # Resample indices
            samp_res_id = np.random.normal(mu, sigma, len(qqx))
            # Plot
            sns.regplot(x=qqx, y=sorted(samp_res_id), lowess=True, ci=False,
            scatter=False,
            line_kws={'color': 'lightgrey', 'lw': 1, 'alpha': 0.8})
            
    sns.regplot(x=qqx, y=qqy, scatter=True, lowess=False, ci=False,
    scatter_kws={'s': 40, 'alpha': 0.5},
    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    axes.plot(qqx, qqx, '--k', alpha=0.5)
    axes.set_title('Normal Q-Q')
    axes.set_xlabel('Theoretical Quantiles')
    axes.set_ylabel('Standardized Residuals')
    
""" Scale-Location Plot """
def plot_scale_loc(axes, yfit, res_stand_sqrt, n_samp=0):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    yfit: fitted/predicted y values
    res_stand_sqrt: Absolute square root Residuals
    n_samp[optional]: number of resamples """
    # For every random resampling
    for i in range(n_samp):
        # 1. resample indices from sqrt Residuals
        samp_res_id = random.sample(list(res_stand_sqrt), len(res_stand_sqrt))
        # 2. Average of Residuals, smoothed using LOWESS
        sns.regplot(x=yfit, y=samp_res_id,
        scatter=False, ci=False, lowess=True,
        line_kws={'color': 'lightgrey', 'lw': 1, 'alpha': 0.8})
        # 3. Repeat again for n_samples
        
    # plot Regression using Seaborn
    sns.regplot(x=yfit, y=res_stand_sqrt,
        scatter=True, ci=False, lowess=True,
    scatter_kws={'s': 40, 'alpha': 0.5},
    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    axes.set_title('Scale-Location plot')
    axes.set_xlabel('Fitted Sales values')
    axes.set_ylabel('$\sqrt{\|Standardized\ Residuals\|}$')
    

def plot_cooks(axes, res_inf_leverage, res_standard, x_lim=None, y_lim=None):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    res_inf_leverage: Leverage
    res_standard: standardized residuals
    x_lim, y_lim[optional]: axis limits """

    sns.regplot(x=res_inf_leverage, y=res_standard,
                scatter=True, ci=False, lowess=True,
                scatter_kws={'s': 40, 'alpha': 0.5},
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    # Set limits
    if x_lim != None:
        x_min, x_max = x_lim[0], x_lim[1]
    else:
        x_min, x_max = min(res_inf_leverage), max(res_inf_leverage)
    if y_lim != None:
        y_min, y_max = y_lim[0], y_lim[1]
    else:
        y_min, y_max = min(res_standard), max(res_standard)
    # Plot centre line
    plt.plot((x_min, x_max), (0, 0), 'g--', alpha=0.8)
    # Plot contour lines for Cook's Distance levels
    n = 100
    cooks_distance = np.zeros((n, n))
    x_cooks = np.linspace(x_min, x_max, n)
    y_cooks = np.linspace(y_min, y_max, n)
    for xi in range(n):
        for yi in range(n):
            cooks_distance[yi][xi] = \
            y_cooks[yi]**2 * x_cooks[xi] / (2 * (1 - x_cooks[xi]))
            
    CS = axes.contour(x_cooks, y_cooks, cooks_distance, levels=4, alpha=0.6)
    axes.clabel(CS, inline=0, fontsize=10)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_title('Residuals vs Leverage and Cook\'s distance')
    axes.set_xlabel('Leverage')
    axes.set_ylabel('Standardized Residuals')

    
def get_residual_plots(fitted_model, fig=plt.figure(figsize = (14,9)), resampling=True, num_samples=100, cooks=False, only_cooks=False):
    """ Summary: Get the residual plots for the fitted model     
    Args:
        fitted_model: fitted model from statsmodels
    Returns:
    plt
    """
    yfit = fitted_model.fittedvalues       # Predictions: 
    res = fitted_model.resid               # Residuals:
    res_inf = fitted_model.get_influence() # Influence:
    res_standard = res_inf.resid_studentized_internal # Standardized residuals
    res_stand_sqrt = np.sqrt(np.abs(res_standard)) # # Absolute square root Residuals:): 
    res_inf_leverage = res_inf.hat_matrix_diag
    
    if only_cooks == True:
        ax1 = fig.add_subplot(1, 1, 1)
        plot_cooks(ax1, res_inf_leverage, res_standard)
        plt.tight_layout()
        return fig
    
    if cooks == True: w = 4
    else: w = 3
    
    if resampling == True: h = 2
    else: h = 1

    # Residuals vs. Fitted values
    ax1 = fig.add_subplot(h, w, 1)
    plot_residuals(ax1, res, yfit)

    # QQ-plot
    ax2 = fig.add_subplot(h, w, 2)
    plot_QQ(ax2, res_standard)

    # Scale-location
    ax3 = fig.add_subplot(h, w, 3)
    plot_scale_loc(ax3, yfit, res_stand_sqrt)

    if cooks == True: 
        ax4 = fig.add_subplot(h, w, 4)
        plot_cooks(ax4, res_inf_leverage, res_standard)

    if resampling == True:
        # Residuals vs Fitted values with 100 resamples
        ax5 = fig.add_subplot(h, w, w+1)
        plot_residuals(ax5, res, yfit, n_samp = num_samples)

        # QQ Plot with 100 resamples
        ax6 = fig.add_subplot(h, w, w+2)
        plot_QQ(ax6, res_standard, n_samp = num_samples)

        # Scale-location with 100 resamples
        ax7 = fig.add_subplot(h, w, w+3)
        plot_scale_loc(ax7, yfit, res_stand_sqrt, n_samp = num_samples)

    # Show plot
    plt.tight_layout()
    return fig


