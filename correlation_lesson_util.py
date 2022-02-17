import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def plot_stripplot_and_sample(ax, df, sample):
    sns.stripplot(data=df, y='student_type', x='grade', color='black', alpha=.05, ax=ax)
    sns.stripplot(
        data=sample, y='student_type', x='grade', ax=ax,
    )
    t, p = stats.ttest_ind(
        sample[sample.student_type == 'Virtual'].grade,
        sample[sample.student_type == 'In-Person'].grade
    )
    ax.set_title(f'p = {p:.4f}')
    return ax

def viz_student_grades_df(df):
    sns.stripplot(data=df, y='student_type', x='grade', color='black', alpha=.5).set_title('Actual Grade Distribution')

    fig, axs = plt.subplots(4, 4, figsize=(24, 14), sharex=True, sharey=True, facecolor='white')
    for ax in axs.ravel()[:-1]:
        sample = df.sample(25)
        plot_stripplot_and_sample(ax, df, sample)

    plot_stripplot_and_sample(axs.ravel()[-1], df, df.sample(25, random_state=503))
    fig.tight_layout()


def viz_ttest_h0_is_true():
    np.random.seed(13)

    df = pd.DataFrame()
    df['grade'] = np.random.normal(70, 10, 500)
    df['student_type'] = np.random.choice(['Virtual', 'In-Person'], 500)
    viz_student_grades_df(df)
    return df

def viz_ttest_h0_is_false():
    df = pd.DataFrame()
    df['student_type'] = np.random.choice(['Virtual', 'In-Person'], 500)
    df['grade'] = np.where(
        df.student_type == 'Virtual',
        np.random.normal(70, 10, 500),
        np.random.normal(80, 10, 500),
    )

    viz_student_grades_df(df)
    return df

def scatterplot_df_and_sample(ax, df, sample):
    r, p = stats.pearsonr(sample.x, sample.y)
    ax.set_title(f'r = {r:.4f}, p = {p:.4f}')
    ax.scatter(df.x, df.y, alpha=.3, color='orange')
    ax.scatter(sample.x, sample.y, color='blue')
    return ax

def viz_pearson_h0_is_true():
    np.random.seed(123)

    df = pd.DataFrame()
    df['x'] = np.random.normal(100, 10, 1000)
    df['y'] = np.random.normal(100, 10, 1000)

    fig, axs = plt.subplots(4, 4, figsize=(24, 14), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax in axs[:-1]:
        scatterplot_df_and_sample(ax, df, df.sample(20))

    scatterplot_df_and_sample(axs[-1], df, df.sample(20, random_state=383))
    fig.suptitle('$H_0$ is true')

    return df

def viz_pearson_h0_is_false():
    np.random.seed(123)

    df = pd.DataFrame()
    df['x'] = np.random.normal(100, 10, 1000)
    df['y'] = df.x + np.random.normal(0, 15, 1000)

    fig, axs = plt.subplots(4, 4, figsize=(24, 14), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax in axs:
        scatterplot_df_and_sample(ax, df, df.sample(20))

    fig.suptitle('$H_0$ is false')
    return df
