# Predicting Track Popularity on Spotify
by Leo Evancie

_[Spotify](https://www.spotify.com/us/) hosts some 70 million songs and caters to over 300 million listeners worldwide. Each track has an associated popularity score, derived from listener engagement. So, Spotify can determine on a post hoc basis whether a given song performs well on the platform. But what about new music? Using the data science method, can we use song characteristics like duration, lyrical explicitness -- even abstract concepts like “danceability” -- to predict whether a song will be popular? Spotify could use such predictions to guide decisions about which new music to promote._

## 1. Data Wrangling
_Code notebook found [here](https://github.com/levancie/spotify-popularity/blob/main/notebooks/1-Data-Wrangling.ipynb)._

Using the [`Spotipy`](https://github.com/plamere/spotipy) library, I scraped 10,000 tracks from the Spotify API -- one thousand tracks from each of the past ten years. From among the myriad available fields, I pulled several empirical features (e.g., track duration, tempo, song name) as well as Spotify's calculations of several less tangible musical characteristics (e.g., 'danceability', 'instrumentalness', and 'acousticness', each ranging from 0.00 to 1.00). The target, popularity, is an integer ranging from 0 to 100.

While the very highest popularity ratings tended to occur in tracks from the most recent year, each of the ten years showed bimodal distributions, with several tracks clustering around zero and the rest clustered around 60-80. The bimodality of popularity across years lent itself well to a binary representation of popularity, where any track with a popularity score greater than 50 could be considered popular. Thus, rather than attempting to predict a precise popularity score via regression, I would instead attempt to classify popular vs. unpopular.

## 2. Exploratory Data Analysis
_Code notebook found [here](https://github.com/levancie/spotify-popularity/blob/main/notebooks/2-Exploratory-Data-Analysis.ipynb)._

I inspected each feature for its relationship to popularity. I found that:
* lower `track numbers` tend to be more popular (both due to the popularity of singles, which generally have a track number of 1, and due to the relative popularity of a given album's earlier tracks)
* `singles` tend to be more popular than tracks belonging to an album or compilation
* tracks with higher `'danceability'` tend to be more popular
* tracks with medium `'energy'` levels are popular, while very low or high 'energy' tracks are not
* a modest cluster of tracks with very high `'instrumentalness'` scores have very low popularity
* 86% of tracks with `explicit` lyrics are popular, and 89% of tracks featuring a `guest artist` are popular
* track `duration` is important to popularity, peaking at durations of about 200 seconds
* tracks with a standard `time signature` (i.e., rhythmic structure) tend to be popular more than those with less conventional structures

## 3. Preprocessing
_Code notebook found [here](https://github.com/levancie/spotify-popularity/blob/main/notebooks/3-Preprocessing.ipynb)._

I completed much of the cleaning and preprocessing as I worked through the first two notebooks, and Spotify's data is kept in good condition to begin with (_very_ few missing values). At this stage, I only needed to prepare three features for machine learning legibility: time signature, track number, and duration.

Time signature being a categorical variable, I simply created dummy columns with Pandas' `.get_dummies()` method.

Track number is a quantitative variable, which would typically require scaling or normalizing to be handled by most machine learning algorithms. However, I could not rescale the values as is, because the vast majority of track numbers fell at or below 25, there was a thin tail reaching up over 100. Rescaling would 'squish' the majority of the values, making it harder for the algorithm to parse the relationship between that feature and the others. Luckily, those tracks with a track number higher than 25 tended to be unpopular. And so, I converted all over-25 track numbers to 25, safe in the knowledge that this would not only preserve the more important part of the distribution of track numbers, but that I would not be losing too much information about how the high track numbers affected popularity. The resulting column was then rescaled from 0 to 1 with the scikit-learn `MinMaxScaler()`.

Track duration, measured in seconds, was rescaled straightforwardly.

## 4. Modeling
_Code notebook found [here](https://github.com/levancie/spotify-popularity/blob/main/notebooks/4-Modeling.ipynb)._

With training data accounting for 70% of my original 10,000-track dataset, I tested four of `scikit-learn`'s most popular machine learning algorithms for classification: `LogisticRegression()`, `KNeighborsClassifier()`, `RandomForestClassifier()`, and `GradientBoostingClassifier`. I employed hyperparameter tuning for each, and recorded the wall-time required to both train the model and generate predictions from the test set.

Overall, I chose the gradient boosting classifier as the best model overall. Like the other ensemble model (random forest), the gradient boosting classifier yielded strong f1 scores. But what set it apart from the random forest were the lesser degree of overfit to the training data, and especially the drastically shorter prediction times.

###### This is a capstone project for [Springboard's](https://www.springboard.com/) Data Science Career Track. Thank you to Mukesh Mithrakumar for your mentorship, to Blaine Bateman for the many project reviews, and to [Paul Lemere](https://github.com/plamere) for the Spotipy library. Leo Evancie, 2021.
