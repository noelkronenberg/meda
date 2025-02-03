import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import statsmodels.api as sm
from stepmix import StepMix
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import LogLocator

import warnings
from sklearn.exceptions import ConvergenceWarning

from . import logger

def logit(data: pd.DataFrame, outcome: str, confounders: list, categorical_vars: list = None, drop_first: bool = True,
          dropna: bool = False, show_results: bool = False, show_forest_plot: bool = False, reference_col: str = None, 
          selected_confounders: list = None, confounder_names: dict = None, custom_colors: list = None, 
          error_bar_colors: list = None) -> sm.Logit:
    """
    Fits a `statsmodels <https://www.statsmodels.org/stable/index.html>`_ logistic regression model to the given data. 
    Optionally plots a forest plot of the odds ratios.

    Args:
        data (pd.DataFrame): The input data containing the outcome variable and confounders.
        outcome (str): The name of the outcome variable column.
        confounders (list): A list of confounders column names to be used in the model.
        dropna (bool, optional): Whether to drop rows with missing values. Defaults to False.
        categorical_vars (list, optional): A list of categorical variable column names to be converted to dummy variables. Defaults to None.
        drop_first (bool, optional): Whether to drop the first dummy variable for each categorical variable or reference_col, when given and applicable. Defaults to True.
        show_results (bool, optional): Whether to print the summary of the logistic regression results. Defaults to False.
        show_forest_plot (bool, optional): Whether to plot a forest plot of the odds ratios. Defaults to False.
        reference_col (str, optional): The reference column for adjusting odds ratios. Defaults to None.
        selected_confounders (list, optional): A list of selected confounders to be included in the forest plot. Defaults to None.
        confounder_names (dict, optional): A dictionary mapping original confounder names to display names in the forest plot. Defaults to None.
        custom_colors (list, optional): A list of custom colors for the points in the forest plot. Defaults to None.
        error_bar_colors (list, optional): A list of custom colors for the error bars in the forest plot. Defaults to None.

    Returns:
        sm.Logit: The fitted logistic regression model.

    Examples:
        >>> import pandas as pd
        >>> from utils.analysis import logit
        >>> data = pd.DataFrame({
        ...     'outcome': [1, 0, 1, 0, 1],
        ...     'confounder_1': [5, 3, 6, 2, 7],
        ...     'confounder_2': [1, 0, 1, 0, 1]
        ... })
        >>> result = logit(
        ...     data=data, 
        ...     outcome='outcome', 
        ...     confounders=['confounder_1', 'confounder_2'], 
        ...     show_forest_plot=True, 
        ...     reference_col='confounder_1'
        ... )
    """
    
    # prepare data
    X = data[confounders]
    y = data[outcome]
    logger.info('Prepared data for logistic regression model.')

    # drop rows with missing values
    if dropna:
        initial_length = len(X)
        missing_data = X.isna().sum()
        X = X.dropna()
        y = y.loc[X.index]
        removed_entries = initial_length - len(X)
        logger.info(f'Dropped {removed_entries} rows with missing values.')
        logger.info(f'Columns with most missing values: {missing_data[missing_data > 0].sort_values(ascending=False).head().to_dict()}')

    if categorical_vars:
        for col in categorical_vars:
            # verify that the columns in categorical_vars are of a categorical data type
            if not pd.api.types.is_categorical_dtype(data[col]):
                data[col] = data[col].astype('category')
                logger.info(f'Converted column {col} to categorical data type.')

        # convert categorical variables to dummy variables
        X = pd.get_dummies(X, columns=categorical_vars, drop_first=False)
        logger.info(f'Converted categorical variables to dummy variables: {categorical_vars} (drop_first=False)')

        # manually drop the first category for each categorical variable, except for the reference column
        if drop_first:
            for col in categorical_vars:
                categories = data[col].cat.categories
                if reference_col and reference_col in X.columns:
                    X.drop(columns=[reference_col], inplace=True)
                    logger.info(f'Dropped reference column: {reference_col}')
                    continue
                dummy_col = f"{col}_{categories[0]}"
                if dummy_col in X.columns:
                    X.drop(columns=[dummy_col], inplace=True)
                    logger.info(f'Dropped first dummy variable column: {dummy_col}')
    
    # ensure binary variables are integers (0 or 1) instead of boolean
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        logger.info(f'Converted boolean column {col} to integer.')
    
    # fit logistic regression model
    model = sm.Logit(y, X)
    result = model.fit(disp=0) # disp=0 suppresses output
    logger.info('Fitted logistic regression model.')

    # ORs and 95% CIs
    odds_ratios = np.exp(result.params)
    conf = np.exp(result.conf_int())
    logger.info('Calculated odds ratios and 95% confidence intervals.')

    # DataFrame for plotting
    or_df = pd.DataFrame({
        'OR': odds_ratios,
        'Lower CI': conf[0],
        'Upper CI': conf[1]
    }).reset_index().rename(columns={'index': 'confounder'})

    # exclude the constant term
    or_df = or_df[or_df['confounder'] != 'const']
    logger.info('Excluded the constant term from the odds ratios.')

    # adjust ORs relative to the reference column
    if reference_col and reference_col in or_df['confounder'].values:
        ref_or = or_df.loc[or_df['confounder'] == reference_col, 'OR'].values[0]
        or_df['OR'] /= ref_or
        or_df['Lower CI'] /= ref_or
        or_df['Upper CI'] /= ref_or
        logger.info(f'Adjusted odds ratios relative to reference column: {reference_col}')

    # filter selected confounders for plotting
    if selected_confounders:
        or_df = or_df[or_df['confounder'].isin(selected_confounders)]
        logger.info(f'Selected confounders for plotting: {selected_confounders}')

    # map original confounder names to display names if confounder_names is provided (as transformed categorical variables cannot be adjusted beforehand
    if confounder_names:
        or_df['confounder'] = or_df['confounder'].map(confounder_names).fillna(or_df['confounder'])
        logger.info(f'Mapped original confounder names to display names: {confounder_names}')

    # plotting
    if show_forest_plot:
        plt.figure(figsize=(10, 6))

        # set x-axis to log scale
        plt.xscale('log')
        logger.info('Set x-axis to log scale.')
        
        # set white background
        plt.gca().set_facecolor('white')
        
        # use custom colors (if provided)
        if error_bar_colors is None:
            error_bar_colors = ['grey'] * len(or_df) # default to grey if not provided
            logger.info('Using default grey color for error bars.')
        
        # create error bars for each confounder
        for i in range(len(or_df)):
            # calculate the error margins
            lower_error = or_df['OR'].iloc[i] - or_df['Lower CI'].iloc[i]
            upper_error = or_df['Upper CI'].iloc[i] - or_df['OR'].iloc[i]
            plt.errorbar(or_df['OR'].iloc[i], or_df['confounder'].iloc[i], 
                         xerr=[[lower_error], [upper_error]],
                         fmt='none', color=error_bar_colors[i], capsize=5)
        logger.info('Created error bars for the odds ratios.')

        # use custom colors if provided
        if custom_colors and len(custom_colors) == len(or_df):
            palette = custom_colors
            logger.info(f'Using custom colors for the points: {custom_colors}')
        else:
            palette = ['blue'] * len(or_df)
            logger.info('Using default blue color for the points.')

        # create a scatter plot
        sns.scatterplot(data=or_df, x='OR', y='confounder', 
                         color='blue', s=100, zorder=2) # default color for the points
        logger.info('Created scatter plot with default colors.')
        
        if error_bar_colors:
            for i in range(len(or_df)):
                plt.scatter(or_df['OR'].iloc[i], or_df['confounder'].iloc[i], 
                            color=palette[i], s=100, zorder=2) # color for each point
            logger.info('Created scatter plot with custom colors.')

        plt.axvline(x=1, color='gray', linestyle='dotted', zorder=1, linewidth=1)
        logger.info('Added vertical line at OR = 1.')

        plt.title('Odds Ratios with 95% Confidence Intervals')
        plt.xlabel('Odds Ratio')
        plt.ylabel('Confounders')
        plt.grid(False)

        # get min and max values for x-axis
        min_x = min(or_df["Lower CI"].min(), or_df["OR"].min())
        max_x = max(or_df["Upper CI"].max(), or_df["OR"].max())
        if not np.isfinite(min_x):
            min_x = or_df["OR"].min()
        if not np.isfinite(max_x):
            max_x = or_df["OR"].max()

        # buffer size for x-axis limits
        buffer_size = 0.1 
        # actual buffer size to apply in log space
        left_buffer = buffer_size
        right_buffer = buffer_size
        # left and right buffers in log scale
        log_min_x = np.log10(min_x) if min_x > 0 else -1
        log_max_x = np.log10(max_x) if max_x > 0 else 0

        # set x-axis limits with the buffer applied in log scale
        plt.xlim(left=10**(log_min_x - left_buffer), right=10**(log_max_x + right_buffer)) 
        logger.info(f'Set x-axis limits: left = {10**(log_min_x - left_buffer)} and right = {10**(log_max_x + right_buffer)}')

        # custom log x-axis ticks
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
        tick_values = np.logspace(log_min_x - left_buffer, log_max_x + right_buffer, num=10)
        # tick_values = np.array([round(v * 2) / 2 if v > 1 else v for v in tick_values]) # round ticks > 1 to .5
        plt.xticks(tick_values, [f'{v:.1f}' for v in tick_values])
        logger.info(f'Set custom log x-axis ticks: {tick_values}')

        plt.show()

    # show results
    if show_results:
        print(result.summary())
        logger.info('Finished displaying logistic regression results.')
    
    return result

def lca(data: pd.DataFrame, outcome: str = None, confounders: list = None, 
        n_classes: list = list(range(1, 11)), fixed_n_classes: int = None, show_metrics: bool = False, cv: int = 3, 
        return_assignments: bool = False, show_polar_plot: bool = False, cmap: str = 'tab10') -> StepMix:
    """
    Fits a Latent Class Analysis (LCA) model to the given data using `StepMix <https://stepmix.readthedocs.io/en/latest/api.html#stepmix>`_. 
    If no outcome or confounders are provided, an unsupervised approach is used.
    Optionally plots a polar plot of the latent class assignments with normalized prevalences.

    Args:
        data (pd.DataFrame): The input data containing the variables for LCA.
        outcome (str, optional): The name of the outcome variable column. Defaults to None.
        confounders (list, optional): A list of confounders column names to be used in the model. Defaults to None.
        n_classes (list, optional): The number of latent classes to fit. Defaults to a range from 1 to 10.
        fixed_n_classes (int, optional): A fixed number of latent classes to use instead of tuning. Defaults to None.
        show_metrics(bool, optional): Whether to plot LCA metrics. Only applies when `fixed_n_classes` is None. Defaults to False.
        cv (int, optional): The number of cross-validation folds for hyperparameter tuning. Defaults to 3.
        return_assignments (bool, optional): Whether to return the latent class assignments for the observations. Defaults to False.
        show_polar_plot (bool, optional): Whether to plot a polar plot of the latent class assignments. Defaults to False.
        cmap (str, optional): The colormap to use for plotting clusters. Defaults to 'tab10'.

    Returns:
        StepMix or tuple: 
            If `return_assignments` is False (default), returns the fitted LCA model (with the highest log likelihood when `n_classes` is used) as a `StepMix` object.
            If `return_assignments` is True, returns a tuple containing:
                - StepMix: The fitted LCA model (with the highest log likelihood when `n_classes` is used).
                - pd.DataFrame: The original data with an additional column named 'latent_class' that contains the predicted latent class assignments for each observation.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.datasets import make_blobs
        >>> from utils.analysis import lca
        >>> # generate synthetic data with 3 actual latent classes
        >>> X, _ = make_blobs(n_samples=1000, centers=3, n_features=6, random_state=42)
        >>> synthetic_data = pd.DataFrame(
        ...     X, 
        ...     columns=['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6']
        ... )
        >>> synthetic_data = (synthetic_data > synthetic_data.median()).astype(int)
        >>> # fit LCA model
        >>> model = lca(data=synthetic_data, n_classes=[2, 3, 4, 5], show_polar_plot=True)
    """

    # imply supervised approach if outcome or confounders are provided
    supervised = outcome and confounders

    # prepare data
    if supervised:
        X = data[confounders]
        y = data[outcome]
        logger.info('Prepared data for supervised LCA model.')
    else:
        logger.info('No outcome or confounders provided. Using unsupervised approach.')

    # ensure columns are binary, warn user if not
    for col in data.columns:
        if data[col].nunique() > 2:
            logger.warning(f'Column {col} is non-binary. LCA results might not be accurate.')

    # base model
    base_model = StepMix(n_components=3, n_steps=1, measurement='bernoulli', structural='bernoulli', random_state=42)

    # hyperparameter tuning or fixed model fitting
    if fixed_n_classes is not None:
        logger.info(f'Using fixed number of latent classes: {fixed_n_classes}.')
        model = StepMix(n_components=fixed_n_classes, n_steps=1, measurement='bernoulli', structural='bernoulli', random_state=42)
        if supervised:
            model.fit(X, y)
        else:
            model.fit(data)
        logger.info(f'Fitted model with {fixed_n_classes} latent classes.')
    else:
        # model selection using hyperparameter tuning
        logger.info(f'Using hyperparameter tuning with {cv} cross-validation folds.')
        gs = GridSearchCV(estimator=base_model, cv=cv, param_grid={'n_components': n_classes}, verbose=0)

        # suppress ConvergenceWarning (as it is expected in LCA with CV)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # fit model
        if supervised:
            gs.fit(X, y)
        else:
            gs.fit(data)
        logger.info(f'Hyperparameter tuning completed with {n_classes} latent classes and {cv} cross-validation folds.')
        
        if show_metrics:
            # calculate additional metrics
            results = pd.DataFrame(gs.cv_results_)
            results['log_likelihood'] = results['mean_test_score']
            bic_values = []
            entropy_values = []
            smallest_class_sizes = []

            # create a new model for each set of parameters
            for params in gs.cv_results_['params']:
                logger.info(f'Calculating additional metrics for model with {params["n_components"]} latent classes.')
                model = StepMix(n_components=params['n_components'], n_steps=1, measurement='bernoulli', structural='bernoulli', random_state=42)

                # fit the model to the data
                if supervised:
                    model.fit(X, y)
                else:
                    model.fit(data)
                logger.info(f'Fitted model with {params["n_components"]} latent classes.')

                # get BIC, relative Entropy, and smallest class size
                if supervised:
                    bic = model.bic(X, y)
                    entropy = model.relative_entropy(X, y)
                    smallest_class_size = min(np.bincount(model.predict(X, y)))
                else:
                    bic = model.bic(data)
                    entropy = model.relative_entropy(data)
                    smallest_class_size = min(np.bincount(model.predict(data)))
                logger.info(f'Calculated additional metrics for model with {params["n_components"]} latent classes.')
                
                # append metrics to lists
                bic_values.append(bic)
                entropy_values.append(entropy)
                smallest_class_sizes.append(smallest_class_size)

            # add metrics to results
            results['BIC'] = bic_values
            results['Entropy'] = entropy_values
            results['Smallest Class Size'] = smallest_class_sizes

            # plot additional metrics

            plt.figure(figsize=(10, 5))
            sns.lineplot(data=results, x='param_n_components', y='log_likelihood', marker='o')
            plt.title('Log Likelihood')
            plt.xlabel('Number of Latent Classes')
            plt.ylabel('Log Likelihood')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            sns.lineplot(data=results, x='param_n_components', y='BIC', marker='o')
            plt.title('BIC')
            plt.xlabel('Number of Latent Classes')
            plt.ylabel('BIC')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            sns.lineplot(data=results, x='param_n_components', y='Entropy', marker='o')
            plt.title('Entropy')
            plt.xlabel('Number of Latent Classes')
            plt.ylabel('Relative Entropy')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            sns.lineplot(data=results, x='param_n_components', y='Smallest Class Size', marker='o')
            plt.title('Smallest Class Size')
            plt.xlabel('Number of Latent Classes')
            plt.ylabel('Smallest Class Size')
            plt.tight_layout()
            plt.show()

            logger.info('Plotted log likelihood, BIC, Entropy, and smallest class size against number of latent classes.')

        model = gs.best_estimator_
        logger.info(f'Best model selected based on hyperparameter tuning: {model}')

    # predict latent class assignments
    if return_assignments or show_polar_plot:

        # copy data to avoid modifying the original
        data_updated = data.copy()

        # predict latent class assignments
        if supervised:
            predictions = model.predict(X, y)
        else:
            predictions = model.predict(data)
        logger.info(f'Predicted {len(predictions)} latent class assignments.')
        
        # add latent class assignments (starting from 1)
        data_updated['latent_class'] = predictions + 1
        logger.info('Merged latent class assignments with observations.')

    # plot polar plot
    if show_polar_plot:

        # use all columns as confounders if not provided
        if not supervised:
            confounders = data.columns.tolist()
            logger.info('Using all columns as confounders for the polar plot in unsupervised LCA model.')

        # calculate prevalence of each latent class
        class_prevalences = data_updated.groupby('latent_class')[confounders].mean().reset_index()
        total_prevalences = data_updated[confounders].mean()
        
        # normalize prevalences
        normalized_prevalences = class_prevalences.copy()
        for confounder in confounders:
            normalized_prevalences[confounder] = class_prevalences[confounder] / total_prevalences[confounder]
        
        logger.info('Calculated normalized prevalence for each confounder in each latent class.')

        # assign latent classes to confounders
        assigned_classes = {}
        for confounder in confounders:

            # get the class with the highest value (if not empty)
            max_value = normalized_prevalences[confounder].max()
            max_classes = normalized_prevalences[normalized_prevalences[confounder] == max_value]['latent_class'].values
            
            # assign class with highest value
            if max_classes.size == 0:
                logger.warning(f'Confounder {confounder} has no classes assigned.')
                assigned_classes[confounder] = None
            else:
                if max_classes.size > 1:
                    logger.warning(f'Confounder {confounder} has multiple classes with the same normalized prevalence. Choosing the first one.')
                max_class = max_classes[0]
                assigned_classes[confounder] = max_class
                logger.info(f'Assigned latent class {max_class} to confounder {confounder} with normalized prevalence {max_value:.4f}.')

        # sort confounders based on assigned classes and prevalence value within each class
        sorted_confounders = []
        for confounder, assigned_class in assigned_classes.items():
            # get the prevalence value for the assigned class
            prevalence_value = normalized_prevalences.loc[normalized_prevalences['latent_class'] == assigned_class, confounder].values[0]
            sorted_confounders.append((confounder, assigned_class, prevalence_value))
        # sort by assigned class and then by prevalence value
        sorted_confounders.sort(key=lambda x: (x[1], -x[2])) 
        # get (only) the sorted confounder names
        sorted_confounder_names = [confounder for confounder, _, _ in sorted_confounders]
        logger.info(f'Sorted confounders based on assigned latent classes and prevalence values: {sorted_confounder_names}')

        # plot polar plot
        fig = go.Figure()
        latent_classes = data_updated['latent_class'].unique()
        colors = sns.color_palette(cmap, n_colors=len(latent_classes)).as_hex()
        for i, latent_class in enumerate(sorted(latent_classes)):

            # filter data for the latent class
            class_data = normalized_prevalences[normalized_prevalences['latent_class'] == latent_class]
            class_values = class_data[sorted_confounder_names].values.flatten()

            # skip if no data available
            if class_data.empty:
                logger.warning(f'No data available for latent class {latent_class}. Skipping.')
                continue
            
            # plot polar plot
            fig.add_trace(go.Scatterpolar(
                r=class_values.tolist() + [class_values[0]], # close the shape
                theta=sorted_confounder_names + [sorted_confounder_names[0]], # close the shape
                name=f'Latent Class {latent_class}', # name for legend
                fill='toself', # fill area inside the shape
                fillcolor=f'rgba({int(int(colors[i][1:3], 16))}, {int(int(colors[i][3:5], 16))}, {int(int(colors[i][5:7], 16))}, 0.1)', # fill color (with transparency)
                line=dict(color=colors[i]), # color for the line
            ))
            logger.info(f'Added polar plot for latent class {latent_class}.')

        # update layout and show figure
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showline=True, linecolor='rgba(0,0,0,0.1)', gridcolor='rgba(0,0,0,0.1)'),
                angularaxis=dict(tickfont=dict(size=24), linecolor='grey', gridcolor='rgba(0,0,0,0.1)'),
                bgcolor='white'
            ),
            showlegend=True,
            legend=dict(font=dict(size=24)),
            width=2400,
            height=1200,
            paper_bgcolor='rgba(255,255,255)',
            plot_bgcolor='rgba(255,255,255)'
        )
        fig.show()

    # return based on parameters
    if return_assignments:
        return model, data_updated
    else:
        return model
