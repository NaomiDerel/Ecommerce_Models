import numpy as np
import pandas as pd
import scipy
import sklearn


def task4(train_df, test_df):
    """
    Implement the naive method of averages from class.
    :param train_df: train data
    :param test_df: test data
    """

    # get uniqu songs and users:
    song_ids = list(set(train_df['song_id']))
    user_ids = list(set(train_df['user_id']))

    # get avergae over all pairs (those that aren't in the data are zero):
    sum_train = train_df['weight'].sum()
    num_cells = len(song_ids) * len(user_ids)
    avg = sum_train/num_cells

    # calculate diviation from avergae per user and song:
    user_avg = train_df.groupby('user_id')['weight'].mean() - avg
    song_avg = train_df.groupby('song_id')['weight'].mean() - avg

    # Calculate function f4:
    f4 = 0
    for index, row in train_df.iterrows():
        # get user and song index:
        user = int(row['user_id'])
        song = int(row['song_id'])

        # the baseline prediction is the avergae:
        pred = avg
        # if the user existed in the train set, add the user bias:
        try:
            pred += user_avg[user]
        except:
            pass
        # if the song existed in the train set, add the song bias:
        try:
            pred += song_avg[song]
        except:
            pass

        # weights are above 0, so set the prediction to be positive:
        f4 += (row['weight'] - max(0, pred)) ** 2

    print(f"f4 = {f4}")

    # Create Csv for test:
    for row_idx, row in test_df.iterrows():
        # get user and song index:
        user = int(row['user_id'])
        song = int(row['song_id'])
        # the baseline prediction is the avergae:
        pred = avg
        # if the user existed in the train set, add the user bias:
        try:
            pred += user_avg[user]
        except:
            pass
        # if the song existed in the train set, add the song bias:
        try:
            pred += song_avg[song]
        except:
            pass
        # weights are above 0, so set the prediction to be positive:
        test_df.at[row_idx, 'weight'] = max(0, pred)

    # transform into csv file:
    test_df = test_df.iloc[:, 1:]
    test_df.to_csv('3_Recommendation_Systems/results/task4.csv', index=False)


def main():
    np.random.seed(0)
    test_df = pd.read_csv("3_Recommendation_Systems/data/test.csv")
    user_song_df = pd.read_csv("3_Recommendation_Systems/data/user_song.csv")

    task4(user_song_df, test_df)


if __name__ == '__main__':
    main()