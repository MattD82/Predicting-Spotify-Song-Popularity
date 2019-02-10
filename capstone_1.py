import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# allows me to plot the scatter matrix
from pandas.plotting import scatter_matrix
# allows me to use train_test_split to get test and train datasets
from sklearn.model_selection import train_test_split

'''
This is my main python script for my first capstone project!

Note I will also be doing quite a bit of analysis in jupyter, just due to the
visual nature of this data. 

All columns in this dataset and a full description of each:
    artist_name (object): Name of the artist.

    track_id (object): Sportify unique track id.

    track_name (object): Name of the track/song.

    acousticness (float): A confidence measure from 0.0 to 1.0 of whether the 
                          track is acoustic. 1.0 represents high confidence 
                          the track is acoustic.

    danceability (float): Danceability describes how suitable a track is for 
                          dancing based on a combination of musical elements 
                          including tempo, rhythm stability, beat strength, 
                          and overall regularity. A value of 0.0 is least 
                          danceable and 1.0 is most danceable.

    duration_ms (int): The duration of the track in milliseconds.
    energy (float): Energy is a measure from 0.0 to 1.0 and represents a 
                    perceptual measure of intensity and activity. Typically, 
                    energetic tracks feel fast, loud, and noisy. For example, 
                    death metal has high energy, while a Bach prelude scores low 
                    on the scale. Perceptual features contributing to this 
                    attribute include dynamic range, perceived loudness, timbre, 
                    onset rate, and general entropy.
    instrumentalness (float): Predicts whether a track contains no vocals. “Ooh” 
                              and “aah” sounds are treated as instrumental in this 
                              context. Rap or spoken word tracks are clearly “vocal”. 
                              The closer the instrumentalness value is to 1.0, the 
                              greater likelihood the track contains no vocal content. 
                              Values above 0.5 are intended to represent instrumental 
                              tracks, but confidence is higher as the value approaches 
                              1.0.
    key
    liveness
    loudness
    mode - major = 1
    speechiness - 
    tempo - beats per minute
    time_signature - the meter of the song or beats per measure (most songs are in 4/4)
    valence
    popularity - overall popularity score (based on # of clicks)
'''

### ALL FUNCTIONS DEFINED HERE ####
### MIGHT THINK OF CONVERTING INTO CLASSES ####
# load in main database of songs and attributes
def load_data():
    df = pd.read_csv('data/SpotifyAudioFeaturesNov2018.csv')
    return df

# set some display options so easier to view all columns at once
def set_view_options(max_cols=50, max_rows=50, max_colwidth=12, dis_width=250):
    pd.options.display.max_columns = max_cols
    pd.options.display.max_rows = max_rows
    pd.set_option('max_colwidth', max_colwidth)
    pd.options.display.width = dis_width

def rename_columns(df):
    #new_col_names = ['WHOA', 'track_id', 'track_name', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence', 'popularity']
    #df = df.rename(columns=new_col_names)
    df.columns = ['artist', 'trk_id', 'trk_name', 'acous', 'dance', 'ms', 
                  'energy', 'instr', 'key', 'live', 'loud', 'mode', 'speech', 
                  'tempo', 't_sig', 'val', 'popularity']
    return df

def get_df_info(df):
    # take an initial look at our data
    print(df.head())

    # take a look at the columns in our data set
    print("The columns are:")
    print(df.columns)

    # look at data types for each
    print(df.info())

    # take a look at data types, and it looks like we have a pretty clean data set!
    # However, I think the 0 popularity scores might throw the model(s) off a bit.
    ''' NEED TO REMEMBER TO TAKE THIS INTO ACCOUNT!!! '''
    print("Do we have any nulls?")
    print(f"Looks like we have {df.isnull().sum().sum()} nulls")

    # let's get a list of all columns, and check the type of that
    all_cols = list(df.columns)
    print(type(df.columns))
    print(type(all_cols))

    # Lets take a look at the average popularity score
    pop_mean = df['popularity'].mean()
    print(pop_mean)

# cool way to truncate the column names to display easier
# can be used with various metrics
def describe_cols(df, L=10):
    '''Limit ENTIRE column width (including header)'''
    # get the max col width
    O = pd.get_option("display.max_colwidth")
    # set max col width to be L
    pd.set_option("display.max_colwidth", L)
    print(df.rename(columns=lambda x: x[:L - 2] + '...' if len(x) > L else x).describe())
    pd.set_option("display.max_colwidth", O)

# How many songs have a popularity score > 90??
# Can I list these songs?
def most_popular_songs(df):
    most_popular = df[df['popularity'] > 90]['popularity'].count()
    print(df[df['popularity'] > 90][['artist_name', 'popularity']])

# plot a scatter plot
def scatter_plot(df, col_x, col_y):
    plt.scatter(df[col_x], df[col_y], alpha=0.2)
    plt.show()

def plot_scatter_matrix(df, num_rows):
    scatter_matrix(df[:num_rows], alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()

def calc_correlations(df, cutoff):
    corr = df.corr()
    print(corr[corr > cutoff])

# get redundant pairs from DataFrame
def get_redundant_pairs(df):
    '''Get diagonal pairs of correlation matrix and all pairs we'll remove (since pair each is doubled in corr matrix)'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            if df[cols[i]].dtype != 'object' and df[cols[j]].dtype != 'object':
                # print("THIS IS NOT AN OBJECT, YO, so you CAN take a corr of it, smarty!")
                pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# get top absolute correlations
def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    
    print("The top absolute correlations are:")
    print(au_corr[0:n])
    return au_corr[0:n]

# Maybe do a quick and dirty first linear regression with all features?
def linear_regression(df):
    X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
          'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
          'speechiness', 'tempo', 'time_signature', 'valence']
    y_col = ['popularity']

    X = df[X_cols]
    y = df[y_col]
    print(X)
    print(y)

if __name__ == "__main__":
    df = load_data()
    set_view_options()
    #df = rename_columns(df)
    get_df_info(df)

    # Take a look at the data with truncated columns
    describe_cols(df, 10)

    # look at top correlations
    get_top_abs_correlations(df, 20)
    