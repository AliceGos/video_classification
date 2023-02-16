#%%
import matplotlib.pyplot as plt


def plot_columns(full_dataset,video_id):
    """ Line plot for first 6 columns of features corresponding to video_id
    Arguments:
    full_dataset: dataset containing features information for all videos
    video_id: video_id for which plots should be shown

    Output: 6 line plots, one per column
    """
    data_video = full_dataset[full_dataset['video_id']==video_id].reset_index(drop=True)
    data_plot = data_video.iloc[:, :6]
    plt.figure()
    # create a plot for each column
    for col in range(data_plot.shape[1]):
        plt.subplot(data_plot.shape[1], 1, col+1)
        plt.plot(data_plot[data_plot.columns[col]])
        plt.ylabel(str(data_plot.columns[col]))
        plt.title('Video '+video_id+ '\n Take off: ' + str(set(data_video['target'])))
        plt.show()