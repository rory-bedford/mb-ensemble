import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# experiment 1
experiment1_multiple = np.load("results/experiment1-multiple.npy")
experiment1_single = np.load("results/experiment1-single.npy")

data = {
    'Epoch': np.arange(len(experiment1_multiple)),
    'Multiple': experiment1_multiple,
    'Single': experiment1_single
}
df = pd.DataFrame(data)

mean_multiple = np.mean(experiment1_multiple)
mean_single = np.mean(experiment1_single)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.lineplot(ax=axes[0], x='Epoch', y='Multiple', data=df)
axes[0].set_title('Multiple Compartments Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_ylim(0, 0.5)
axes[0].yaxis.set_ticks(np.arange(0, 0.6, 0.1))
axes[0].annotate(
    f'Mean Loss: {mean_multiple:.2f}\nPositive Compartments: 8\nNegative Compartments: 7',
    xy=(0.02, -0.3),
    xycoords='axes fraction',
    ha='left',
    fontsize=10,
    color='black'
)

sns.lineplot(ax=axes[1], x='Epoch', y='Single', data=df)
axes[1].set_title('Single Compartments Loss Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_ylim(0, 0.5)
axes[1].yaxis.set_ticks(np.arange(0, 0.6, 0.1))
axes[1].annotate(
    f'Mean Loss: {mean_single:.2f}\nPositive Compartments: 1\nNegative Compartments: 1',
    xy=(0.02, -0.3),
    xycoords='axes fraction',
    ha='left',
    fontsize=10,
    color='black'
)

plt.tight_layout()
plt.savefig('figures/experiment1_loss_curves.png')
plt.close()


# experiment 2
results_df = pd.read_csv("results/experiment2.csv")

best_models = results_df[results_df['activation'] != "linear"].sort_values("total_loss").head(20)

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Prepare data for paired lines plot
shift = 0.02
existing_pairs = {}
for i, row in best_models.iterrows():
    lower_lr = row['lower_lr']
    upper_lr = row['upper_lr']
    pair = (lower_lr, upper_lr)
    if pair in existing_pairs:
        lower_lr += shift * existing_pairs[pair]
        upper_lr += shift * existing_pairs[pair]
        existing_pairs[pair] += 1
    else:
        existing_pairs[pair] = 1
    axes[0].plot(['Lower learning rate', 'Upper learning rate'], [10**lower_lr, 10**upper_lr], marker='o')

axes[0].set_title('Distribution of Learning Rates')
axes[0].set_ylabel('Learning Rate')
axes[0].set_yscale('log')
axes[0].set_ylim(10**-6, 10**1)
axes[0].set_xticks(['Lower learning rate', 'Upper learning rate'])

# Prepare data for Positive and Negative compartments plot
shift = 0.03
existing_pairs = {}
for i, row in best_models.iterrows():
    n_positive = row['n_positive']
    n_negative = row['n_negative']
    pair = (n_positive, n_negative)
    if pair in existing_pairs:
        n_positive += shift * existing_pairs[pair]
        n_negative += shift * existing_pairs[pair]
        existing_pairs[pair] += 1
    else:
        existing_pairs[pair] = 1
    axes[1].plot(['Positive', 'Negative'], [n_positive, n_negative], marker='o')

axes[1].set_title('Number of Positive and Negative Compartments')
axes[1].set_ylabel('Compartments')
axes[1].set_xticks(['Positive', 'Negative'])

fig.suptitle('Best 20 Models', fontsize=16)

fig.text(0.5, 0.0, 'Rate of change: 10 epochs\nFraction of change: 0.1\nGrid search parameters: learning rates, activation functions, number compartments\nModels trained: 1728', ha='center', fontsize=10, color='black')

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('figures/experiment2_plots.png')
plt.close()

# loss curves

experiment2_best = np.load("results/experiment2-bestrun.npy")
experiment2_random = np.load("results/experiment2-run3.npy")

data = {
    'Epoch': np.arange(len(experiment2_best)),
    'Best': experiment2_best,
    'Random': experiment2_random
}
df = pd.DataFrame(data)

mean_best = np.mean(experiment2_best)
mean_random = np.mean(experiment2_random)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.lineplot(ax=axes[0], x='Epoch', y='Best', data=df)
axes[0].set_title('Best Model Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_ylim(0, 0.5)
axes[0].yaxis.set_ticks(np.arange(0, 0.6, 0.1))
axes[0].annotate(
    f'Mean Loss: {mean_best:.2f}',
    xy=(0.02, -0.15),
    xycoords='axes fraction',
    ha='left',
    fontsize=10,
    color='black'
)

sns.lineplot(ax=axes[1], x='Epoch', y='Random', data=df)
axes[1].set_title('Random Model Loss Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_ylim(0, 0.5)
axes[1].yaxis.set_ticks(np.arange(0, 0.6, 0.1))
axes[1].annotate(
    f'Mean Loss: {mean_random:.2f}',
    xy=(0.02, -0.15),
    xycoords='axes fraction',
    ha='left',
    fontsize=10,
    color='black'
)

plt.tight_layout()
plt.savefig('figures/experiment2_loss_curves.png')
plt.close()


# experiment 3
results_df = pd.read_csv("results/experiment3.csv")
losses = results_df['mean_loss'].to_numpy()
losses = losses[losses != 'mean_loss'].astype(np.float32)
losses = losses.tolist()

plt.figure()
plt.scatter(np.ones(len(losses)) + np.random.normal(0,0.03,size=len(losses)), losses, alpha=0.5, color='black', marker='+',zorder=2)
plt.boxplot(losses, vert=True, widths=0.5, patch_artist=True, boxprops=dict(facecolor="lightblue"),zorder=1)
plt.ylabel("Loss")
plt.annotate(
    f'Rates of change: [3, 10, 30] epochs',
    xy=(0.02, -0.1),
    xycoords='axes fraction',
    ha='left',
    fontsize=10,
    color='black'
)
plt.xticks([])
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.title("Distribution of losses")
plt.savefig('figures/experiment3_boxplot.png')