import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as scp
import sklearn


def f1(data, b, r, user_id_mapping, song_id_mapping, num_users):
    """
    Calculate the value of the goal function in question 1 for the train set.
    :param data: train set df
    :param b: user and song biases
    :param r: weight - average_weight for each user
    :param user_id_mapping: translator to indices of users
    :param song_id_mapping:translator to indices of songs
    :param num_users: number of users
    :return: value of f1 function on the data
    """
    sum = 0
    for index, row in data.iterrows():
        error = r[index] - b[user_id_mapping[row['user_id']]] - \
            b[song_id_mapping[row['song_id']] + num_users]
        sum += error ** 2
    return sum


def task1(test_df, user_song_df):
    """
    Preform task 1, print value of f1 function, and save a csv with predictions on test.
    :param test_df: test data
    :param user_song_df: train data
    """
    # get all users and songs ids:
    song_ids = list(set(user_song_df['song_id']).union(
        set(test_df['song_id'])))
    user_ids = list(set(user_song_df['user_id']).union(
        set(test_df['user_id'])))
    column_names = song_ids + user_ids

    # calculate the average rating over the entire data and calculate each weight minus the average:
    r = user_song_df['weight']
    r_avg = user_song_df['weight'].mean()
    r = r - r_avg
    # create a sparse matrix from the data, and mappings to translate indexes:
    sparse_matrix = sps.lil_matrix(
        (len(user_song_df), len(column_names)), dtype=np.int8)
    user_id_mapping = {id: index for index, id in enumerate(user_ids)}
    song_id_mapping = {id: index for index, id in enumerate(song_ids)}

    # init one hot the matrix sparse_matrix according to the known data like learned in class:
    for row_idx, row in user_song_df.iterrows():
        user_idx = user_id_mapping[row['user_id']]
        song_idx = song_id_mapping[row['song_id']]
        sparse_matrix[row_idx, user_idx] = 1  # Set user column to 1
        # Set song column to 1
        sparse_matrix[row_idx, len(user_ids) + song_idx] = 1

    # solve the least square equation tp find optimal biases:
    b = scp.lsqr(sparse_matrix, r)
    b = b[0]

    # Calculate f1 on train:
    f1_val = f1(user_song_df, b, r, user_id_mapping,
                song_id_mapping, len(user_ids))
    print(f"f1 = {f1_val}")
    if f1_val < 950000000000:
        print("f1 passed limit!")

    # Create Csv for test:
    for row_idx, row in test_df.iterrows():
        # can't be negative so set minimum possible value to be 0:
        test_df.at[row_idx, 'weight'] = max(0, r_avg + b[user_id_mapping[row['user_id']]] + b[
            song_id_mapping[row['song_id']] + len(user_ids)])
    test_df = test_df.iloc[:, 1:]
    test_df.to_csv('3_Recommendation_Systems/results/task1.csv', index=False)


def f2(Q, P, user_song_df):
    """
    Calculate the value of the goal function in question 2 for the train set.
    :param Q: songs matrix
    :param P: users matrix
    :param user_song_df: data from the train set
    :return: value of f1 function on the data
    """
    # get predictions:
    PTQ = P.T @ Q
    new_df = []
    for _, row in user_song_df.iterrows():
        user_id = row['user_id']
        song_id = row['song_id']
        new_df.append(PTQ.loc[user_id, song_id])
    new_df = pd.DataFrame(new_df)
    new_df.rename(columns={0: 'weight'}, inplace=True)
    b = pd.DataFrame(user_song_df['weight'])
    # calculate loss function:
    residuals = b.sub(new_df, axis='columns')
    return np.sum(residuals ** 2)


def task2(test_df, user_song_df):
    """
    Preform task 2, print value of f1 function, and save a csv with predictions on test.
    :param test_df: test data
    :param user_song_df: train data
    """

    # get a sorted lists of all users and songs ids:
    song_ids_list = list(
        set(user_song_df['song_id']).union(set(test_df['song_id'])))
    user_ids_list = list(
        set(user_song_df['user_id']).union(set(test_df['user_id'])))
    song_ids_list.sort()
    user_ids_list.sort()

    # initialize a matrix for all users/songs with some random values:
    P = pd.DataFrame(np.random.rand(20, len(user_ids_list)),
                     columns=user_ids_list)
    Q = pd.DataFrame(np.random.rand(20, len(song_ids_list)),
                     columns=song_ids_list)

    # initialize parameters for alternating least square problem:
    is_Q = True
    f2_value = f2(Q, P, user_song_df)['weight']
    previous_f2 = f2_value + 300001
    user_songs_dict = {}
    i = 0
    songs_user_dict = {}

    # create dictionaries so for each song/user we will know what users/songs paring we have for it:
    for user_id, group in user_song_df.groupby('user_id'):
        song_ids = group['song_id'].tolist()
        user_songs_dict[user_id] = song_ids
    for song_id, group in user_song_df.groupby('song_id'):
        user_ids = group['user_id'].tolist()
        songs_user_dict[song_id] = user_ids

    # while we haven't reached our goal, solve the new Q or P with alternating least square method:
    while 300000000 < f2_value or previous_f2 - f2_value < 300000:
        if is_Q:  # Q
            for song_id in songs_user_dict.keys():
                b = np.array(
                    user_song_df.loc[user_song_df['song_id'] == song_id, 'weight'])
                Pu = np.reshape(
                    P[songs_user_dict[song_id]].values, (20, len(b)))
                try:
                    Q.loc[:, song_id] = pd.DataFrame(np.linalg.lstsq(
                        Pu.T, b, rcond=None)[0].T, columns=[song_id])
                except np.linalg.LinAlgError:  # Matrix did not converge:
                    Q.loc[:, song_id] = Q.loc[:, song_id]
        else:  # P
            for user_id in user_songs_dict.keys():
                b = np.array(
                    user_song_df.loc[user_song_df['user_id'] == user_id, 'weight'])
                Qi = np.reshape(
                    Q[user_songs_dict[user_id]].values, (20, len(b)))
                try:
                    P.loc[:, user_id] = pd.DataFrame(np.linalg.lstsq(
                        Qi.T, b.T, rcond=None)[0].T, columns=[user_id])
                except np.linalg.LinAlgError:  # Matrix did not converge:
                    P.loc[:, user_id] = P.loc[:, user_id]

        # update parameter:
        previous_f2 = f2_value
        if i % 20 == 0:  # saves run-time of checks:
            f2_value = f2(Q, P, user_song_df)['weight']
        i += 1
        is_Q = not is_Q

    # Calculate f2 on train:
    print(f"f2 = {f2_value}")
    if f2_value < 300000000:
        print("f2 passed limit!")

    # Create Csv for test:
    for row_idx, row in test_df.iterrows():
        # can't be negative so set minimum possible value to be 0:
        test_df.at[row_idx, 'weight'] = max(
            0, P.loc[:, row['user_id']].T @ Q.loc[:, row['song_id']])
    test_df = test_df.iloc[:, 1:]
    test_df.to_csv('3_Recommendation_Systems/results/task2.csv', index=False)


def task3(df, test_df):
    """
    Preform task 3, print value of f1 function, and save a csv with predictions on test.
    :param df: train data
    :param test_df: test data
    :return:
    """

    # parameters:
    k = 20
    users = df["user_id"].unique()
    songs = df["song_id"].unique()
    shape = (len(users), len(songs))

    # Create indices for users and movies:
    user_cat = pd.api.types.CategoricalDtype(
        categories=sorted(users), ordered=True)
    song_cat = pd.api.types.CategoricalDtype(
        categories=sorted(songs), ordered=True)
    user_index = df["user_id"].astype(user_cat).cat.codes
    song_index = df["song_id"].astype(song_cat).cat.codes

    # Conversion via COO matrix:
    A = scipy.sparse.coo_matrix(
        (df['weight'], (user_index, song_index)), shape=shape).astype(float).tocsr()
    U, s, V = scipy.sparse.linalg.svds(A, k=k)

    # calculate avg:
    song_ids = list(set(df['song_id']))
    user_ids = list(set(df['user_id']))
    sum_train = df['weight'].sum()
    num_cells = len(song_ids) * len(user_ids)
    avg = sum_train / num_cells

    # Prediction on test to csv file:
    with open("3_Recommendation_Systems/results/task3.csv", 'w') as csvfile:
        csvfile.write("user_id,song_id,weight\n")

        for index, row in test_df.iterrows():
            try:
                # find new index for the current user:
                user = int(row['user_id'])
                song = int(row['song_id'])
                new_user_index = user_cat.categories.get_loc(user)
                new_song_index = song_cat.categories.get_loc(song)

                # predict weight for user, can't be negative so set minimum possible value to be 0:
                pred = max(0, U[new_user_index].T @
                           np.diag(s) @ V[:, new_song_index])

            # in case the user or song didn't exist in the train (which shouldn't happen according to the instructions),
            # predict the avg so the program keeps running:
            except:
                pred = avg
            csvfile.write(str(user) + "," + str(song) + "," + str(pred) + "\n")

    # Calculate function f3:
    f3 = 0
    for index, row in df.iterrows():
        user = int(row['user_id'])
        song = int(row['song_id'])
        new_user_index = user_cat.categories.get_loc(user)
        new_song_index = song_cat.categories.get_loc(song)

        pred = max(0, U[new_user_index].T @ np.diag(s) @ V[:, new_song_index])
        f3 += (row['weight'] - pred) ** 2

    print(f"f3 = {f3}")
    if f3 < 235000000000:
        print("f3 passed limit!")


def main():
    np.random.seed(0)
    test_df = pd.read_csv("3_Recommendation_Systems/data/test.csv")
    user_song_df = pd.read_csv("3_Recommendation_Systems/data/user_song.csv")

    task1(test_df, user_song_df)
    task2(test_df, user_song_df)
    task3(user_song_df, test_df)


if __name__ == '__main__':
    main()
