import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from config import features_PATH  
from config import images_PATH  
from config import models_PATH  
from config import recordings_PATH  
def make_eda_plots():
    fig_dims = (10, 7)
    fig, ax = plt.subplots(figsize=fig_dims)

    df_features_path = os.path.join(features_PATH, 'df_features.csv')
    df = pd.read_csv(df_features_path)

    plot_emotions = sns.countplot(
        x="emotion", data=df, color="lightseagreen", ax=ax
    ).set_title("RAVDESS Audio Dataset")
    plot_emotions_path = os.path.join(images_PATH, 'plot_emotions.png')
    plot_emotions.figure.savefig(plot_emotions_path)
    plot_intensity = sns.countplot(
        x="intensity", data=df, color="lightseagreen", ax=ax
    ).set_title("RAVDESS Audio Dataset")
    plot_intensity_path = os.path.join(images_PATH, 'plot_intensity.png')
    plot_intensity.figure.savefig(plot_intensity_path)
    plot_gender = sns.countplot(
        x="gender", data=df, color="lightseagreen", ax=ax
    ).set_title("RAVDESS Audio Dataset")
    
    plot_gender_path = os.path.join(images_PATH, 'plot_gender.png')
    plot_gender.figure.savefig(plot_gender_path)
    print("Successfully created plots.")


if __name__ == "__main__":
    make_eda_plots()
