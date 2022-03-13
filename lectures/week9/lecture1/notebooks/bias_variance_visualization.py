# Import 3rd party libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Configure Notebook
plt.style.use('fivethirtyeight')
sns.set_context("notebook")
import warnings
warnings.filterwarnings('ignore')


def func(x):
    return -1.0 + 0.003 * x + 0.05 * x**2 + 0.003 * x**3


def fit_model(deg, ml_model, train, test=None):

    # Tranform data
    trans = PolynomialFeatures(degree=deg)
    x_train = trans.fit_transform(train['x'].to_numpy().reshape(-1, 1))
    if test is not None:
        x_test = trans.transform(test['x'].to_numpy().reshape(-1, 1))
    else:
        x_test = None

    # Fit model
    model = ml_model.fit(x_train, train['y'].to_numpy().reshape(-1, 1))

    # Get predictions
    x_plotting = np.linspace(-10, 10, 1000).reshape(-1, 1)
    y_plotting = model.predict(trans.transform(x_plotting))
    y_train_pred = model.predict(x_train)
    if test is not None:
        y_test_pred = model.predict(x_test)
    else:
        y_test_pred = None

    return x_plotting, y_plotting, y_train_pred, y_test_pred, model


def bias_variance_visualizer(data, deg, test_split, n_samples, n_models, folder, ml_model):

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    y_train_preds = []
    y_test_preds = []
    x_model_plot = []
    y_model_plot = []
    train_rmses = []
    test_rmses = []
    models = []

    for _ in range(n_models):
        # Get sample
        data_sample = data.sample(n_samples)

        # Split data
        train, test = train_test_split(data_sample, test_size=test_split)

        # Fit model
        x_plotting, y_plotting, y_train_pred, y_test_pred, model = fit_model(deg, ml_model, train, test)

        # Get data
        x_trains.append(train[['x']].to_numpy())
        y_trains.append(train[['y']].to_numpy())
        x_tests.append(test[['x']].to_numpy())
        y_tests.append(test[['y']].to_numpy())
        y_train_preds.append(y_train_pred)
        y_test_preds.append(y_test_pred)
        x_model_plot.append(x_plotting)
        y_model_plot.append(y_plotting)
        models.append(model)
        train_rmses.append(np.sqrt(mean_squared_error(train[['y']].to_numpy(), y_train_pred)))
        test_rmses.append(np.sqrt(mean_squared_error(test[['y']].to_numpy(), y_test_pred)))

    rmses = pd.DataFrame({'Train': train_rmses, 'Test': test_rmses})

    # Create directory
    os.makedirs(os.path.join(os.path.abspath(os.getcwd()), folder), exist_ok=True)

    for idx in range(n_models):

        # Setup plot
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(wspace=0.4)
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((1, 5), (0, 4))

        """Train"""
        ax1.set_title('Model: Polynomial Order {}\nTrain (80%)'.format(deg), fontsize=26, loc='left', y=1.05)

        # Plot true function
        sns.lineplot(np.arange(-10, 10, 0.01), func(np.arange(-10, 10, 0.01)),
                     color='#fc4f30', ax=ax1, lw=1, label='True Function')

        # Population
        sns.scatterplot(x='x', y='y', data=data, ax=ax1, label='Population', color=[0.7, 0.7, 0.7], alpha=0.25)

        # Plot current model and data
        if idx > 0:
            for prev_idx in range(0, idx):
                ax1.plot(x_model_plot[prev_idx].flatten(), y_model_plot[prev_idx].flatten(), color='#6d904f', lw=1,
                         alpha=0.75)

        # Plot current model and data
        ax1.vlines(x_trains[idx].flatten(), ymin=y_trains[idx].flatten(), ymax=y_train_preds[idx].flatten(),
                   linestyle='-', color='#e5ae38', alpha=0.3, zorder=0, label='Error')
        sns.scatterplot(x_trains[idx].flatten(), y_trains[idx].flatten(), ax=ax1, label='Sample', s=60)
        ax1.plot(x_model_plot[idx].flatten(), y_model_plot[idx].flatten(), label='Prediction', color='#6d904f', lw=4)

        # Format figure
        ax1.xaxis.set_tick_params(labelsize=16)
        ax1.yaxis.set_tick_params(labelsize=16)
        ax1.set_xlabel('x', fontsize=22)
        ax1.set_ylabel('y', fontsize=22)
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-3, 8])
        ax1.legend(loc=2, fontsize=16)

        """Test"""
        ax2.set_title('Test (20%)', fontsize=26, loc='left', y=1.05)

        # Plot true function
        sns.lineplot(np.arange(-10, 10, 0.01), func(np.arange(-10, 10, 0.01)),
                     color='#fc4f30', ax=ax2, lw=1, label='True Function')

        # Population
        sns.scatterplot(x='x', y='y', data=data, ax=ax2, label='Population', color=[0.7, 0.7, 0.7], alpha=0.25)

        # Plot current model and data
        if idx > 0:
            for prev_idx in range(0, idx):
                ax2.plot(x_model_plot[prev_idx].flatten(), y_model_plot[prev_idx].flatten(), color='#6d904f', lw=1,
                         alpha=0.75)

        # Plot current model and data
        ax2.vlines(x_tests[idx].flatten(), ymin=y_tests[idx].flatten(), ymax=y_test_preds[idx].flatten(), linestyle='-',
                   color='#e5ae38', alpha=0.3, zorder=0, label='Error')
        sns.scatterplot(x_tests[idx].flatten(), y_tests[idx].flatten(), ax=ax2, label='Sample', s=60)
        ax2.plot(x_model_plot[idx].flatten(), y_model_plot[idx].flatten(), label='Prediction', color='#6d904f', lw=4)

        # Format figure
        ax2.xaxis.set_tick_params(labelsize=16)
        ax2.yaxis.set_tick_params(labelsize=16)
        ax2.set_xlabel('x', fontsize=22)
        ax2.set_ylabel('y', fontsize=22)
        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-3, 8])
        ax2.legend(loc=2, fontsize=16)

        """Performance"""
        sns.violinplot(data=rmses.iloc[0:idx + 1, :], ax=ax3, color="white", scale='width')
        sns.swarmplot(data=rmses.iloc[0:idx + 1, :], ax=ax3, s=20)
        # ax3.set_ylim([0, None])
        ax3.xaxis.set_tick_params(labelsize=22)
        ax3.yaxis.set_tick_params(labelsize=16)
        ax3.set_ylabel('RMSE', fontsize=22)

        plt.savefig(os.path.join(os.path.abspath(os.getcwd()), folder, str(idx) + '.png'))
