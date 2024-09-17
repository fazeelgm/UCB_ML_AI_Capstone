import matplotlib.pyplot as plt
import seaborn as sns


# Global declarations
page_width = 9

sns.set_theme(style='darkgrid')

def graph_features(df, nrows, ncols, figsize=(7,7)):
    """
    Graph all the DataFrame columns as subplots to quickly visualize the features 
    :param df: DataFrame to iterate on
    :param nrows: Number of rows in subplot
    :param ncols: Number of cols in subplot
    :param fisize: list containing (widht, height)
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()
    
    # Loop through each column and plot a histogram
    for i, column in enumerate(df.columns):
        
        # Add the histogram
        df[column].hist(ax=axes[i], # Define on which ax we're working on
                        edgecolor='white', # Color of the border
                        color='#69b3a2' # Color of the bins
                       )
        
        # Add title and axis label
        axes[i].set_title(f'{column}') 
        axes[i].set_xlabel(column) 
        axes[i].set_ylabel('Frequency') 
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()