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
    
def get_residual_plots(fitted_model, fig):
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


    # Residuals vs. Fitted values
    ax1 = fig.add_subplot(2, 3, 1)
    plot_residuals(ax1, res, yfit)

    # QQ-plot
    ax2 = fig.add_subplot(2, 3, 2)
    plot_QQ(ax2, res_standard)

    # Scale-location
    ax3 = fig.add_subplot(2, 3, 3)
    plot_scale_loc(ax3, yfit, res_stand_sqrt)

    # Residuals vs Fitted values with 100 resamples
    ax4 = fig.add_subplot(2, 3, 4)
    plot_residuals(ax4, res, yfit, n_samp = 100)

    # QQ Plot with 100 resamples
    ax5 = fig.add_subplot(2, 3, 5)
    plot_QQ(ax5, res_standard, n_samp = 100)

    # Scale-location with 100 resamples
    ax6 = fig.add_subplot(2, 3, 6)
    plot_scale_loc(ax6, yfit, res_stand_sqrt, n_samp = 100)

    # Show plot
    plt.tight_layout()
    plt.show()