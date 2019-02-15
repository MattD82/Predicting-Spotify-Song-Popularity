import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import random

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats  
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import itertools

'''
This is the main python script for my first capstone project.

The ultimate goal of this script was to use linear and logistic regression
in order to predict a popularity score, using the attributes listed below
as dependent variables.

All columns in this dataset and a full description of each.
Note these descriptions are from Spotify's API website.
    artist_name (object): Name of the artist.

    track_id (object): Spotify unique track id.

    track_name (object): Name of the track/song.

    1 - acousticness (float): A confidence measure from 0.0 to 1.0 of whether the 
                          track is acoustic. 1.0 represents high confidence 
                          the track is acoustic.

    2 - danceability (float): Danceability describes how suitable a track is for 
                          dancing based on a combination of musical elements 
                          including tempo, rhythm stability, beat strength, 
                          and overall regularity. A value of 0.0 is least 
                          danceable and 1.0 is most danceable.

    3 - duration_ms (int): The duration of the track in milliseconds.
    4 - energy (float): Energy is a measure from 0.0 to 1.0 and represents a 
                    perceptual measure of intensity and activity. Typically, 
                    energetic tracks feel fast, loud, and noisy. For example, 
                    death metal has high energy, while a Bach prelude scores low 
                    on the scale. Perceptual features contributing to this 
                    attribute include dynamic range, perceived loudness, timbre, 
                    onset rate, and general entropy.
    5 - instrumentalness (float): Predicts whether a track contains no vocals. “Ooh” 
                              and “aah” sounds are treated as instrumental in this 
                              context. Rap or spoken word tracks are clearly “vocal”. 
                              The closer the instrumentalness value is to 1.0, the 
                              greater likelihood the track contains no vocal content. 
                              Values above 0.5 are intended to represent instrumental 
                              tracks, but confidence is higher as the value approaches 
                              1.0.
    6 - key (int): The estimated overall key of the track. Integers map to pitches using 
               standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. 
               If no key was detected, the value is -1.
    7 - liveness (float): Detects the presence of an audience in the recording. 
                        Higher liveness values represent an increased probability that the 
                        track was performed live. A value above 0.8 provides strong likelihood 
                        that the track is live.
    8 - loudness: (float): The overall loudness of a track in decibels (dB). 
        Loudness values are averaged across the entire track and are useful for comparing 
        relative loudness of tracks. Loudness is the quality of a sound that is the primary 
        psychological correlate of physical strength (amplitude). Values typical range between 
        -60 and 0 db.
    9 - mode( int): Mode indicates the modality (major or minor) of a track, the type of scale 
        from which its melodic content is derived. Major is represented by 1 and minor is 0.
    10 - speechiness (float): Speechiness detects the presence of spoken words in a track. 
        The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), 
        the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably 
        made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain 
        both music and speech, either in sections or layered, including such cases as rap music. 
        Values below 0.33 most likely represent music and other non-speech-like tracks. 
    11 - tempo - beats per minute
    12 - time_signature (int): An estimated overall time signature of a track. The time signature 
    (meter) is a notational convention to specify how many beats are in each bar (or measure).
    13 - valence - (float): A measure from 0.0 to 1.0 describing the musical positiveness conveyed by 
    a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks 
    with low valence sound more negative (e.g. sad, depressed, angry).
    popularity - overall popularity score (based on # of clicks) The popularity of the track. The value 
    will be between 0 and 100, with 100 being the most popular.

The popularity of a track is a value between 0 and 100, with 100 being the most popular. 
The popularity is calculated by algorithm and is based, in the most part, on the total number of plays 
the track has had and how recent those plays are.
Generally speaking, songs that are being played a lot now will have a higher popularity than songs that 
were played a lot in the past.

'''

### Define Global Variables ###
global object_cols
object_cols = ['artist_name', 'track_id', 'track_name', 'key_notes','pop_cat']

global numeric_cols
numeric_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness',
        'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence',
       'popularity','pop_frac','pop_bin']

global categorical_cols
categorical_cols = ['key', 'mode', 'time_signature']

global numeric_non_cat
numeric_non_cat = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
       'loudness', 'speechiness', 'tempo','valence',
       'popularity','pop_frac','pop_bin']

global cols_to_stardardize
cols_to_standardize = ['duration_ms', 'loudness', 'tempo']

### ALL FUNCTIONS DEFINED HERE ####
# load in main database of songs and attributes
def load_data():
    df = pd.read_csv('data/SpotifyAudioFeaturesNov2018.csv')
    return df

# set some display options so easier to view all columns at once
def set_view_options(max_cols=50, max_rows=50, max_colwidth=9, dis_width=250):
    pd.options.display.max_columns = max_cols
    pd.options.display.max_rows = max_rows
    pd.set_option('max_colwidth', max_colwidth)
    pd.options.display.width = dis_width

# allows for easier visualization of all columns at once in the terminal
def rename_columns(df):
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
    print("Do we have any nulls?")
    print(f"Looks like we have {df.isnull().sum().sum()} nulls")

    # Lets take a look at the average popularity score
    pop_mean = df['popularity'].mean()
    print(pop_mean)

    # Proportion of songs that are very popular
    print(df[df['popularity'] >= 50 ]['popularity'].count() / df.shape[0])

    # Unique artists and song counts by artist
    print(df['artist_name'].unique().shape)
    print(df['artist_name'].value_counts())

# nice way to truncate the column names to display easier
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
# Let's list these songs
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
    '''Get diagonal pairs of correlation matrix and all pairs we'll remove 
    (since pair each is doubled in corr matrix)'''
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

# initial linear regression function, and plots
def linear_regression_initial(df):
    df = df.copy()

    X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
          'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
          'speechiness', 'tempo', 'time_signature', 'valence']

    y_col = ['popularity']

    X = df[X_cols]
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train = sm.add_constant(X_train)

    # Instantiate OLS model, fit, predict, get errors
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    fitted_vals = results.predict(X_train)
    stu_resid = results.resid_pearson
    residuals = results.resid
    y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, \
                           'stu_resid': stu_resid})

    # Print the results
    print(results.summary())

    # QQ Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title("QQ Plot - Initial Linear Regression")
    fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)
    plt.show()

    # Residuals Plot
    y_vals.plot(kind='scatter', x='fitted_vals', y='stu_resid')
    plt.show()

# print count of all zeros within the dataset
def get_zeros(df):
    print(df[df['popularity'] == 0 ]['popularity'].count())

# plot polularity scores distribution
def plot_pop_dist(df):
    # set palette
    sns.set_palette('muted')

    # create initial figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    sns.distplot(df['popularity']/100, color='g', label="Popularity").set_title("Distribution of Popularity Scores - Entire Data Set")

    # create x and y axis labels
    plt.xlabel("Popularity")
    plt.ylabel("Density")

    plt.show()

# plot undersampling methodology
def undersample_plot(df):
    # set palette
    sns.set_palette('muted')

    # create initial figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    sns.distplot(df['popularity']/100, color='g', label="Popularity").set_title("Illustration of Undersampling from Data Set")
    
    # create line to shade to the right of
    line = ax.get_lines()[-1]
    x_line, y_line = line.get_data()
    mask = x_line > 0.55
    x_line, y_line = x_line[mask], y_line[mask]
    ax.fill_between(x_line, y1=y_line, alpha=0.5, facecolor='red')

    # get values for and plot first label
    label_x = 0.5
    label_y = 4
    arrow_x = 0.6
    arrow_y = 0.2

    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "First, sample all songs in this range.\n Sample size is n. Cutoff is 0.5.", xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
        arrowprops=arrow_properties)

    # Get values for and plot second label
    label_x = 0.1
    label_y = 3
    arrow_x = 0.2
    arrow_y = 0.2

    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "Next, randomly sample \n n songs in this range", xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='g', alpha=0.5),
        arrowprops=arrow_properties)

    # plot final word box
    plt.annotate(
        "Therefore, end up with a 50/50 \n split of Popular / Not Popular\n songs", xy=(0.6, 2),
        xytext=(0.62, 2),
        bbox=dict(boxstyle='round,pad=0.5', fc='b', alpha=0.5))

    # create x and y axis labels
    plt.xlabel("Popularity")
    plt.ylabel("Density")

    plt.show()

# calculate and print more stats from the df
def get_stats(df):
    # print stats for various metrics
    print(f"There are {df.shape[0]} rows")
    print(f"There are {df['track_id'].unique().shape} unique songs")
    print(f"There are {df['artist_name'].unique().shape} unique artists")
    print(f"There are {df['popularity'].unique().shape} popularity scores")
    print(f"The mean popularity score is {df['popularity'].mean()}")
    print(f"There are {df[df['popularity'] > 55]['popularity'].count()} songs with a popularity score > 55")
    print(f"There are {df[df['popularity'] > 75]['popularity'].count()} songs with a popularity score > 75")
    print(f"Only {(df[df['popularity'] > 80]['popularity'].count() / df.shape[0])*100:.2f} % of songs have a popularity score > 80")

# plot univariate dists for several independent variables
def plot_univ_dists(df, cutoff):
    popularity_cutoff = cutoff
    print('Mean value for Danceability feature for Popular songs: {}'.format(df[df['popularity'] > popularity_cutoff]['danceability'].mean()))
    print('Mean value for Danceability feature for Unpopular songs: {}'.format(df[df['popularity'] < popularity_cutoff]['danceability'].mean()))
    
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    fig.suptitle('Histograms and Univariate Distributions of Important Features')
    sns.distplot(df[df['popularity'] < popularity_cutoff]['danceability'])
    sns.distplot(df[df['popularity'] > popularity_cutoff]['danceability'])
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.distplot(df[df['popularity'] < popularity_cutoff]['valence'])
    sns.distplot(df[df['popularity'] > popularity_cutoff]['valence'])
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.distplot(df[df['popularity'] < popularity_cutoff]['acousticness'])
    sns.distplot(df[df['popularity'] > popularity_cutoff]['acousticness'])
    plt.show()

# plot violin plot for several independent variables
def plot_violin(df, cutoff):
    df = df.copy()
    
    sns.set(style="whitegrid")
    df['pop_bin'] = np.where(df['popularity'] > cutoff, "Popular", "Not_Popular")
    
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12,4))
    fig.suptitle('Distributions of Selected Features at Popularity Score Cutoff of 55')
    
    sns.violinplot(x=df['pop_bin'], y=df['danceability'], ax=ax[0])
    sns.violinplot(x=df['pop_bin'], y=df['valence'], ax=ax[1])
    sns.violinplot(x=df['pop_bin'], y=df['acousticness'], ax=ax[2])
    
    plt.show()

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12,4))
    fig.suptitle('Distributions of Selected Features at Popularity Score Cutoff of 55')

    sns.violinplot(x=df['pop_bin'], y=df['energy'], ax=ax[0])
    sns.violinplot(x=df['pop_bin'], y=df['instrumentalness'], ax=ax[1])
    sns.violinplot(x=df['pop_bin'], y=df['liveness'], ax=ax[2])
    
    plt.show()

# plot pairplot for subsection of df rows and columns
def plot_pairplot(df, rows, cutoff):
    # not it looks MUCH better to run this function in jupyter
    df = df.copy()
    
    df['pop_bin'] = np.where(df['popularity'] > cutoff, "Popular", "Not_Popular")
    
    cols_for_pp = ['danceability', 'energy', 'instrumentalness',
       'loudness','valence', 'popularity', 'pop_bin']

    sns.pairplot(df.loc[:rows, cols_for_pp], hue='pop_bin', size=2)

    plt.show()

# plot the key counts for popular and unpopular songs
def plot_keys(df, cutoff):
    df_popular = df[df['popularity'] > cutoff].copy()
    
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8,5))
    key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 4.0: 'E', 5.0: 
                  'F', 6.0: 'F♯,G♭', 7.0: 'G', 8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 
                  11.0: 'B'}
    
    df_popular['key_val'] = df_popular['key'].map(key_mapping)
    sns.countplot(x='key_val', data=df_popular, order=df_popular['key_val'].value_counts().index, palette='muted')
    plt.title("Key Totals for Popular Songs")
    plt.show()

    df_unpopular = df[df['popularity'] < 55].copy()
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8,5))
    df_unpopular['key_val'] = df_unpopular['key'].map(key_mapping)
    sns.countplot(x='key_val', data=df_unpopular, order=df_unpopular['key_val'].value_counts().index, palette='muted')
    plt.title("Key Totals for Unpopular Songs")
    plt.show()

# plot a heatmap of the correlations between features as well as dependent variable
def plot_heatmap(df):
    # note this looks better in jupyter as well
    plt.figure(figsize = (16,6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, )
    plt.show()

# check that deltas in means are significant for selected dependent variables
def calc_ANOVA(df, cutoff):
    df_popular = df[df['popularity'] > cutoff].copy()
    df_unpopular = df[df['popularity'] < cutoff].copy()

    print("Popular and Unpopular Danceability Means:")  
    print(df_popular['danceability'].mean())
    print(df_unpopular['danceability'].mean())
    f_val, p_val = stats.f_oneway(df_popular['danceability'], df_unpopular['danceability'])  
    
    print("Danceability One-way ANOVA P ={}".format(p_val)) 

    print("Popular and Unpopular Loudness Means:")  
    print(df_popular['loudness'].mean())
    print(df_unpopular['loudness'].mean())
    f_val, p_val = stats.f_oneway(df_popular['loudness'], df_unpopular['loudness'])  
    
    print("Loudness One-way ANOVA P ={}".format(p_val)) 

    print(df_popular['valence'].mean())
    print(df_unpopular['valence'].mean())
    f_val, p_val = stats.f_oneway(df_popular['valence'], df_unpopular['valence'])  
    
    print("Valence One-way ANOVA P ={}".format(p_val))

    print(df_popular['instrumentalness'].mean())
    print(df_unpopular['instrumentalness'].mean())
    f_val, p_val = stats.f_oneway(df_popular['instrumentalness'], df_unpopular['instrumentalness'])  
    
    print("Instrumentalness One-way ANOVA P ={}".format(p_val))

# randomly sample data below cutoff after choosing a cutoff so have a 50/50 split
# of popular/unpopular target variable values.
def random_under_sampler(df, cutoff):
    df_original = df.copy()
    df_original['pop_bin'] = np.where(df_original['popularity'] > cutoff, "Popular", "Not_Popular")

    df_small = df_original[df_original['popularity'] > cutoff].copy()
    df_samples_added = df_small.copy()
    
    total = df_small.shape[0] + 1

    # loop through and add random unpopular rows to sampled df
    while total <= df_small.shape[0]*2:

        # pick a random from from the original dataframe
        rand_row = random.randint(0,df_original.shape[0])
        
        if df_original.loc[rand_row, 'pop_bin'] == "Not_Popular":
            df_samples_added.loc[total] = df_original.loc[rand_row, :]
            total +=1

    # print some stats on the undersampled df
    print("Size checks for new df:")
    print("Shape of new undersampled df: {}".format(df_samples_added.shape))
    print(df_samples_added['pop_bin'].value_counts())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Popular']['danceability'].mean())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Not_Popular']['danceability'].mean())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Popular']['danceability'].count())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Not_Popular']['danceability'].count())
    f_val, p_val = stats.f_oneway(df_samples_added[df_samples_added['pop_bin'] == 'Popular']['danceability'], df_samples_added[df_samples_added['pop_bin'] == 'Not_Popular']['danceability'])  
  
    print("One-way ANOVA P ={}".format(p_val))

    # return the df
    return df_samples_added

# plot histograms of metrics for popular and unpopular songs
def plot_hist(sampled_df):
    sampled_df[sampled_df['pop_bin'] == "Popular"].hist(figsize=(8, 8))  
    plt.show()

    sampled_df[sampled_df['pop_bin'] != "Popular"].hist(figsize=(8, 8))
    plt.show()

# return records that contain strings of artist and track names
def search_artist_track_name(df, artist, track):
    # this displays much better in jupyter
    print(df[(df['artist_name'].str.contains(artist)) & (df['track_name'].str.contains(track))])

    # use this if searching for A$AP rocky (or other artist with $ in the name)
    # df[(df['artist_name'].str.contains("A\$AP Rocky"))]

# add important columns to dataframe
def add_cols(df, cutoff=55):
    df = df.copy()
    
    # add key_notes mapping key num vals to notes
    key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 
                   4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 
                   8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}
    df['key_notes'] = df['key'].map(key_mapping)
    
    # add columns relating to popularity
    df['pop_frac'] = df['popularity'] / 100
    df['pop_cat'] = np.where(df['popularity'] > cutoff, "Popular", "Not_Popular")
    df['pop_bin'] = np.where(df['popularity'] > cutoff, 1, 0)
    
    return df

# choose cutoff, sample popular data, randomly sample unpopular data, and combine the dfs
def split_sample_combine(df, cutoff=55, col='popularity', rand=None):
    # split out popular rows above the popularity cutoff
    split_pop_df = df[df[col] > cutoff].copy()
    
    # get the leftover rows, the 'unpopular' songs
    df_leftover = df[df[col] < cutoff].copy()
    
    # what % of the original data do we now have?
    ratio = split_pop_df.shape[0] / df.shape[0]
    
    # what % of leftover rows do we need?
    ratio_leftover = split_pop_df.shape[0] / df_leftover.shape[0]
    
    # get the exact # of unpopular rows needed, using a random sampler
    unpop_df_leftover, unpop_df_to_add = train_test_split(df_leftover, \
                                                          test_size=ratio_leftover, \
                                                          random_state = rand)
    
    # combine the dataframes to get total rows = split_pop_df * 2
    # ssc stands for "split_sample_combine"
    ssc_df = split_pop_df.append(unpop_df_to_add).reset_index(drop=True)

    # shuffle the df
    ssc_df = ssc_df.sample(frac=1, random_state=rand).reset_index(drop=True)
    
    # add key_notes mapping key num vals to notes
    key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 
                   4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 
                   8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}
    ssc_df['key_notes'] = ssc_df['key'].map(key_mapping)
    
    # add columns relating to popularity
    ssc_df['pop_frac'] = ssc_df['popularity'] / 100
    ssc_df['pop_cat'] = np.where(ssc_df['popularity'] > cutoff, "Popular", "Not_Popular")
    ssc_df['pop_bin'] = np.where(ssc_df['popularity'] > cutoff, 1, 0)
    
    return ssc_df

# standardize data and return X and y dfs for linear regresssion
def standardize_return_X_y(df, std=True, log=False):
    df = df.copy()
    
    # standardize some columns if std = True
    if std == True:
        for col in cols_to_standardize:
            new_col_name = col + "_std"
            df[new_col_name] = (df[col] - df[col].mean()) / df[col].std()

        X_cols = ['acousticness', 'danceability', 'duration_ms_std', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness_std', 'mode', 
                  'speechiness', 'tempo_std', 'time_signature', 'valence']
    else:
        X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
                  'speechiness', 'tempo', 'time_signature', 'valence']
        
    # if log = True, let's transform y to LOG
    if log == True:
        df['pop_log'] = df['popularity'] / 100
        df['pop_log'] = [0.00000001 if x == 0 else x for x in df['pop_log']]
        df['pop_log'] = [0.99999999 if x == 1 else x for x in df['pop_log']]
        df['pop_log'] = np.log(df['pop_log'] / (1 - df['pop_log']))
        y_col = ['pop_log']
            
    else:
        y_col = ['popularity']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]
    
    return X, y

# final, clean, linear regression function
def linear_regression_final(df, show_plots=True):
    X, y = standardize_return_X_y(df, std=True, log=False)

    # Add constant
    X = sm.add_constant(X)

    # Instantiate OLS model, fit, predict, and get errors
    model = sm.OLS(y, X)
    results = model.fit()
    fitted_vals = results.predict(X)
    stu_resid = results.resid_pearson
    residuals = results.resid
    y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, \
                           'stu_resid': stu_resid})

    # Maybe do a line graph for this?
    print(results.summary())
    
    ### Plot predicted values vs. actual/true
    if show_plots == True:
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.title("True vs. Predicted Popularity Values - Initial Linear Regression")
        plt.plot(y,alpha=0.2, label="True")
        plt.plot(fitted_vals,alpha=0.5, c='r', label="Predicted")
        plt.ylabel("Popularity")
        plt.legend()
        plt.show()

    # QQ Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)
        plt.show()
  

    # Residuals Plot
        y_vals.plot(kind='scatter', y='fitted_vals', x='stu_resid')
        plt.show()

    return results

# calculate root mean squared error
def my_rmse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return np.sqrt(mse)

# create a linear regression using sklearn, in order to compare models, and 
# also incorporate train_test_split into this, and calculate and print RMSE
def linear_regression_sklearn(df, show_plots=True):
    X, y = standardize_return_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Fit model using the training set
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    # Call predict to get the predicted values for training and test set
    train_predicted = linear.predict(X_train)
    test_predicted = linear.predict(X_test)

    # Calculate RMSE for training and test set
    print('RMSE for training set {}'.format(my_rmse(y_train.values, train_predicted)))
    print('RMSE for test set {}'.format(my_rmse(y_test.values, test_predicted)))
    print('The Coefficients are:')
    print(linear.coef_)
    print('The R^2 values is: {}'.format(linear.score(X_train, y_train)))

    if show_plots == True:
        plt.plot(y_train.reset_index(drop=True), alpha=0.2)
        plt.plot(train_predicted, alpha=0.5, c='r')
        plt.show()

        plt.plot(y_test.reset_index(drop=True), alpha=0.2)
        plt.plot(test_predicted, alpha=0.5, c='r')
        plt.show()

# various data standardization and X/y split functions for logisitic reression
# based on the columns you want to standardize and return
def return_X_y_logistic(df):
    df = df.copy()

    # define columns to use for each
    X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
              'speechiness', 'tempo', 'time_signature', 'valence']

    # use 1's and 0's for logistic
    y_col = ['pop_bin']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]

    return X, y

def return_X_y_logistic_more_cols(df):
    df = df.copy()

    # define columns to use for each
    X_cols = ['artist_name','track_id','track_name','acousticness', 'danceability', 'duration_ms', 'energy', 
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
              'speechiness', 'tempo', 'time_signature', 'valence']

    # use 1's and 0's for logistic
    y_col = ['pop_bin']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]

    return X, y

def return_X_y_logistic_sig_only(df):
    df = df.copy()

    # define columns to use for each
    X_cols = ['danceability','energy', 
              'instrumentalness', 'loudness']

    # use 1's and 0's for logistic
    y_col = ['pop_bin']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]

    return X, y

def standardize_X_sig_only(X):  
    X = X.copy()
    
    cols = ['loudness']
    # standardize only columns not between 0 and 1
    for col in cols:
        new_col_name = col + "_std"
        X[new_col_name] = (X[col] - X[col].mean()) / X[col].std()
        
    X_cols = ['danceability','energy', 
              'instrumentalness', 'loudness_std']

    # return the std columns in a dataframe
    X = X[X_cols]
    
    return X

def standardize_X(X):  
    X = X.copy()
    
    # standardize only columns not between 0 and 1
    for col in cols_to_standardize:
        new_col_name = col + "_std"
        X[new_col_name] = (X[col] - X[col].mean()) / X[col].std()
        
    X_cols = ['acousticness', 'danceability', 'duration_ms_std', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness_std', 'mode', 
                  'speechiness', 'tempo_std', 'time_signature', 'valence']

    # return the std columns in a dataframe
    X = X[X_cols]
    
    return X

def standardize_X_train_test(X_train, X_test):  
    X_train = X_train.copy()
    X_test = X_test.copy() 
    
    # standardize only columns not between 0 and 1
    for col in cols_to_standardize:
        new_col_name = col + "_std"
        X_train[new_col_name] = (X_train[col] - X_train[col].mean()) / X_train[col].std()
        X_test[new_col_name] = (X_test[col] - X_test[col].mean()) / X_test[col].std()
    
    X_cols = ['acousticness', 'danceability', 'duration_ms_std', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness_std', 'mode', 
                  'speechiness', 'tempo_std', 'time_signature', 'valence']

    # return the std columns in a dataframe
    X_train_std = X_train[X_cols]
    X_test_std = X_test[X_cols]
    
    return X_train_std, X_test_std

# Create a basic logistic regression
def basic_logistic_regression(df, cutoff=55, rand=0, sig_only=False):
    df = df.copy()

    if sig_only == True:
        X, y = return_X_y_logistic_sig_only(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X_sig_only(X)

    else:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=80, rand=rand))
        X = standardize_X(X)

    X_const = add_constant(X, prepend=True)

    logit_model = Logit(y, X_const).fit()
    
    print(logit_model.summary())

    return logit_model

def logistic_regression_with_kfold(df, cutoff=55, rand=0, sig_only=False):
    df = df.copy()
    
    if sig_only == True:
        X, y = return_X_y_logistic_sig_only(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X_sig_only(X)

    else:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X(X)

    X = X.values
    y = y.values.ravel()

    classifier = LogisticRegression()

    # before kFold
    y_predict = classifier.fit(X, y).predict(X)
    y_true = y
    accuracy_score(y_true, y_predict)
    print(f"accuracy: {accuracy_score(y_true, y_predict)}")
    print(f"precision: {precision_score(y_true, y_predict)}")
    print(f"recall: {recall_score(y_true, y_predict)}")
    print(f"The coefs are: {classifier.fit(X,y).coef_}")

    # with kfold
    kfold = KFold(len(y))

    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kfold:
        model = LogisticRegression()
        model.fit(X[train_index], y[train_index])

        y_predict = model.predict(X[test_index])
        y_true = y[test_index]

        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    print(f"accuracy: {np.average(accuracies)}")
    print(f"precision: {np.average(precisions)}")
    print(f"recall: {np.average(recalls)}")

# this is the code for the final logistic regression I chose, after running all the above
# logistic regression models and k-fold cross-val analysis
def logistic_regression_final(df, plot_the_roc=True):
    df = df.copy()
    cutoff = 80
    
    X, y = return_X_y_logistic_more_cols(split_sample_combine(df, cutoff=cutoff, rand=2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

    global df_train_results_log80 
    global df_test_results_log80
    df_train_results_log80 = X_train.join(y_train)
    df_test_results_log80 = X_test.join(y_test)

    # standardize X_train and X_test
    X_train = standardize_X(X_train)
    X_test = standardize_X(X_test)

    X_train = X_train.values
    y_train = y_train.values.ravel()

    X_test = X_test.values
    y_test = y_test.values.ravel()

    global sanity_check
    sanity_check = X_test

    ## Run logistic regression on all the data
    classifier = LogisticRegression()
    # note using .predict_proba() below, which is the probability of each class
    
    #predict values for X_train
    y_predict_train = classifier.fit(X_train,y_train).predict(X_train)
    probs_0and1_train = classifier.fit(X_train,y_train).predict_proba(X_train)
    y_prob_P_train = probs_0and1_train[:,1]

    # predict values for X_test
    y_predict_test = classifier.fit(X_train,y_train).predict(X_test)
    probs_0and1_test = classifier.fit(X_train,y_train).predict_proba(X_test) # yes!
    y_prob_P_test = probs_0and1_test[:,1]

    # calculate metrics needed to use for ROC curve below
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_prob_P_train, pos_label=1)
    auc_train = metrics.roc_auc_score(y_train, y_prob_P_train) # note we are scoring on our training data!

    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_prob_P_test, pos_label=1)
    auc_test = metrics.roc_auc_score(y_test, y_prob_P_test) # note we are scoring on our training data!

    # print some metrics
    print("Train accuracy: {:.2f}".format(accuracy_score(y_train, y_predict_train)))
    print("Test accuracy: {:.2f}".format(accuracy_score(y_test, y_predict_test)))

    print("Train recall: {:.2f}".format(recall_score(y_train, y_predict_train)))
    print("Test recall: {:.2f}".format(recall_score(y_test, y_predict_test)))

    print("Train precision: {:.2f}".format(precision_score(y_train, y_predict_train)))
    print("Test precision: {:.2f}".format(precision_score(y_test, y_predict_test)))

    print("Train auc: {:.2f}".format(auc_train))
    print("Test auc: {:.2f}".format(auc_test))

    global conf_matrix_log80_train
    global conf_matrix_log80_test
    conf_matrix_log80_train = confusion_matrix(y_train, y_predict_train)
    conf_matrix_log80_test = confusion_matrix(y_test, y_predict_test)

    global final_coefs
    global final_intercept
    final_coefs = classifier.fit(X_train,y_train).coef_
    final_intercept = classifier.fit(X_train,y_train).intercept_

    # Back of the envelope calcs to make sure metrics above are correct
    df_train_results_log80 = df_train_results_log80.reset_index(drop=True)
    df_train_results_log80['pop_predict'] = y_prob_P_train

    df_test_results_log80 = df_test_results_log80.reset_index(drop=True)
    df_test_results_log80['pop_predict'] = y_prob_P_test

    df_train_results_log80['pop_predict_bin'] = np.where(df_train_results_log80['pop_predict'] >= 0.5, 1, 0)
    df_test_results_log80['pop_predict_bin'] = np.where(df_test_results_log80['pop_predict'] >= 0.5, 1, 0)
    
    print("Back of the envelope calc for Train Recall")
    print(sum((df_train_results_log80['pop_predict_bin'].values * df_train_results_log80['pop_bin'].values))/ df_train_results_log80['pop_bin'].sum())

    if plot_the_roc == True:
        # Plot the ROC
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
                label='Luck')
        ax.plot(fpr_train, tpr_train, color='b', lw=2, label='Model_Train')
        ax.plot(fpr_test, tpr_test, color='r', lw=2, label='Model_Test')
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.set_title("ROC curve - Cutoff: " + str(cutoff), fontsize=24)
        ax.text(0.05, 0.95, " ".join(["AUC_train:",str(auc_train.round(3))]), fontsize=20)
        ax.text(0.32, 0.7, " ".join(["AUC_test:",str(auc_test.round(3))]), fontsize=20)
        ax.legend(fontsize=24)
        plt.show()

# print out confusion matrix
def print_confusion_matrix(df, cutoff=55, rand=0):
    df = df.copy()

    X, y = return_X_y_logistic(split_sample_combine(df, cutoff=80, rand=rand))
    X = standardize_X(X)

    X = X.values
    y = y.values.ravel()

    ## Run logistic regression on all the data
    classifier = LogisticRegression()
    # note using .predict() below, which uses default 0.5 for a binary classifier
    y_pred = classifier.fit(X,y).predict(X) # agh! this uses 0.5 threshold for binary classifier
    y_true = y

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    print("| TN | FP |\n| FN | TP |\n")
    print(cnf_matrix)
    print(f"The accurracy is {accuracy_score(y_true, y_pred)}")
    print(f"The accurracy (check) is {(cnf_matrix[1][1]+ cnf_matrix[0][0])/np.sum(cnf_matrix)}")

# plot popularity score cutoffs vs. logistic regression metrics
def plot_cutoffs_vs_metrics(df):
    df = df.copy()

    df_cols = ['auc', 'accuracy', 'precision', 'recall', 'cutoff', 'type']
    df_metrics = pd.DataFrame(columns = df_cols)
    cutoff_range = [45, 55, 60, 65, 70, 75, 80, 85, 90]
    
    for cutoff in cutoff_range:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=cutoff, rand=0))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        X_train = standardize_X(X_train)
        X_test = standardize_X(X_test)

        X_train = X_train.values
        y_train = y_train.values.ravel()

        X_test = X_test.values
        y_test = y_test.values.ravel()
        
        classifier = LogisticRegression()
        y_predict_train = classifier.fit(X_train, y_train).predict(X_train)
        probs_0and1_train = classifier.fit(X_train,y_train).predict_proba(X_train)
        y_prob_P_train = probs_0and1_train[:,1]
        
        test_metrics = []
        # calculate metrics for JUST train
        test_metrics.append(metrics.roc_auc_score(y_train, y_prob_P_train))
        test_metrics.append(accuracy_score(y_train, y_predict_train))
        test_metrics.append(precision_score(y_train, y_predict_train))
        test_metrics.append(recall_score(y_train, y_predict_train))
        test_metrics.append(int(cutoff))
        test_metrics.append("Test")
        
        df_metrics.loc[cutoff] = test_metrics
        df_metrics = df_metrics.reset_index(drop=True)
        df_metrics["cutoff"] = pd.to_numeric(df_metrics["cutoff"])
        
    # plot metrics vs. popularity score cutoff
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    ax.plot(df_metrics['cutoff'], df_metrics['auc'], color='b', lw=2, label='auc')
    ax.plot(df_metrics['cutoff'], df_metrics['accuracy'], color='r', lw=2, label='accuracy')
    ax.plot(df_metrics['cutoff'], df_metrics['precision'], color='g', lw=2, label='precision')
    ax.plot(df_metrics['cutoff'], df_metrics['recall'], color='y', lw=2, label='recall')

    ax.set_xlabel("Popularity Score Cutoff", fontsize=20)
    ax.set_ylabel("Area (auc) / Rate (others)", fontsize=20)
    ax.set_title("Metrics vs Popularity Score Cutoff Values - Training Dataset:", fontsize=24)
    ax.legend(fontsize=24)
    plt.show()

# plot a confusion matrix
def plot_confusion_matrix(cm, ax, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    """
    font_size = 24
    p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title,fontsize=font_size)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=16)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=16)
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == 1 and j == 1:
            lbl = "(True Positive)"
        elif i == 0 and j == 0:
            lbl = "(True Negative)"
        elif i == 1 and j == 0:
            lbl = "(False Negative)"
        elif i == 0 and j == 1:
            lbl = "(False Positive)"
        ax.text(j, i, "{:0.2f} \n{}".format(cm[i, j], lbl),
                 horizontalalignment="center", size = font_size,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    ax.set_ylabel('True',fontsize=font_size)
    ax.set_xlabel('Predicted',fontsize=font_size)

# plot confusion matrix for final Train dataset
def plot_conf_matrix_Train():
    fig = plt.figure(figsize=(12,11))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ["Not Popular","Popular"]
    plot_confusion_matrix(conf_matrix_log80_train, ax, classes=class_names,normalize=True,
                      title='Normalized Confusion Matrix, Train Dataset, threshold = 0.5')
    plt.show()

# plot confusion matrix for final Test dataset
def plot_conf_matrix_Test():
    fig = plt.figure(figsize=(12,11))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ["Not Popular","Popular"]
    plot_confusion_matrix(conf_matrix_log80_test, ax, classes=class_names,normalize=True,
                      title='Normalized Confusion Matrix, Test Dataset, threshold = 0.5')
    plt.show()

# plot final coefficients of logistic regression
def plot_final_coeffs():
    columns_bar = ['acousticness', 'danceability','duration_ms', 'energy', 'instrumentalness', 
                   'key', 'liveness','loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 
                   'valence']
    df_final_coefs = pd.DataFrame(data = final_coefs, columns = columns_bar)
    df_final_coefs.plot(kind = 'bar', figsize=(10, 5), align='edge')
    plt.show()

def get_true_positives():
    # Songs my test model predicted were popular that are actually popular (true positives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 1) & (df_test_results_log80['pop_bin'] == 1)])

def get_true_negatives():
    # Songs my test model predicted were not popular that are not actually popular (true negatives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 0) & (df_test_results_log80['pop_bin'] == 0)])

def get_false_positives():
    # Songs my testodel predicted were popular that are not actually popular (false positives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 1) & (df_test_results_log80['pop_bin'] == 0)])
    # calculate false positive rate
    df_train_results_log80[(df_train_results_log80['pop_predict_bin'] == 1) & (df_train_results_log80['pop_bin'] == 0)].count() / df_train_results_log80[df_train_results_log80['pop_bin'] == 0].count()

def get_false_negatives():
    # Songs my test model predicted were not popular that are actually popular (false negatives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 0) & (df_test_results_log80['pop_bin'] == 1)])

def sanity_check_test():
    # grab a record from the results dataframe
    sanity_check_loc = df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 0) & (df_test_results_log80['pop_bin'] == 1)].iloc[0]
    # set the probability that song has a popularity score >=80 = sanity_check_prob
    sanity_check_prob = sanity_check_loc['pop_predict']

    # print these to make sure they make sense
    print(sanity_check_loc)
    print(sanity_check_prob)

    # this record coresponds to the 9th row of X_test within the logistic regression function (I know becuase I looked ;)
    print(sanity_check[9, :])
    
    # grab the standardized variables from X_test
    sanity_check_std_vars = sanity_check[9, :]
    print(sanity_check_std_vars)

    # multiply the standardized variables by the regression coefficients, sum them and add the intercept
    mult_coefs_vars_add_intercept = sum(sanity_check_std_vars*final_coefs.reshape(13)) + final_intercept
    print(mult_coefs_vars_add_intercept)

    # since the log odds = P / 1-P, need to exponentiate this to get to the final predicted probability
    exponentiated = np.exp(mult_coefs_vars_add_intercept)
    print(exponentiated)

    # finally, calculate P, the odds of popular (popularity score >= 80)
    p = exponentiated / (1 + exponentiated)
    print(p)

    # does this equal what we think it should???
    delta_ps = float(p - sanity_check_prob)
    print(f"Dela in p values is {delta_ps:.7f}, woo hoo!!!")

if __name__ == "__main__":
    # load data
    df = load_data()
    # set nice view options for terminal viewing
    set_view_options(max_cols=50, max_rows=50, max_colwidth=40, dis_width=250)

    ''' All basic functions commented out so tons of plots don't pop up at once'''
    # get basic info from dataset
    #get_df_info(df)

    # Take a look at the data with truncated columns
    #describe_cols(df, 9)

    # look at top correlations - look into multicollinearity
    #get_top_abs_correlations(df, 10)
    
    ''' All these plots are commented out for now '''  
    # scatter_plot(df, 'danceability', 'popularity')
    # scatter_plot(df, 'duration_ms', 'popularity')
    # scatter_plot(df, 'key', 'popularity')
    # scatter_plot(df, 'acousticness', 'popularity')
    
    ''' Uncomment these to run any or all of the functions defined above 
        Many of these are plots, so commented out for now     ''' 
    #linear_regression_initial(df)
    #plot_pop_dist(df)
    #undersample_plot(df)
    #get_stats(df)
    #plot_univ_dists(df, 85)
    #plot_violin(df, 55)
    #plot_pairplot(df, 500, 55)
    #plot_keys(df, 55)
    #plot_heatmap(df)
    #calc_ANOVA(df, 55)
    #df_samples = random_under_sampler(df, 80)
    #linear_regression_initial(df_samples)
    #plot_hist(df_samples)
    #search_artist_track_name(df, "Chain", "Some")
    #df_cols = add_cols(df, 80)
    #df_split = split_sample_combine(df, cutoff=65, rand=0)
    #linear_regression_final(df_split, show_plots=False)
    #linear_regression_sklearn(df_split, show_plots=True)
    #basic_logistic_regression(df, cutoff=80, rand=0)
    #logistic_regression_with_kfold(df, cutoff=80, rand=0)
    #logistic_regression_with_kfold(df, cutoff=80, rand=0, sig_only=True)
    #print_confusion_matrix(df, cutoff=80, rand=0)
    
    # Calculate and plot final logistic regression values
    logistic_regression_final(df, plot_the_roc=False)
    #plot_cutoffs_vs_metrics(df)
    #plot_conf_matrix_Train()
    #plot_conf_matrix_Test()
    print(final_coefs)
    #plot_final_coeffs()
    #get_true_positives()
    #get_true_negatives()
    #get_false_positives()
    #get_false_negatives()
    sanity_check_test()


    





