import json
import pandas as pd
import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # label assigned to the node
        self.label = label

    def is_leaf(self):
        # a node is leaf node if label is assigned
        return self.label is not None


class DecisionTreeModel:

    def __init__(self, max_depth=100, min_samples_split=2, impurity_stopping_threshold=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    # Fit X and y into DT to train the model and build the DT ???
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # convert DataFrame to numpy array
        # call the _fit method
        X_array = X.to_numpy()
        Y_array = y.to_numpy()
        print("Start fitting")
        self._fit(X_array, Y_array)

    # Build DT
    def _fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # X is a 2D array, each row is a sample, each column is a feature
        self.n_samples, self.n_features = X.shape
        # Unique of y is unique labels of the classifier
        self.n_class_labels = len(np.unique(y))
        # print("********** Depth # ***********", depth,
        #       ", # samples: ", self.n_samples,
        #       ", # features: ", self.n_features,
        #       ", # labels: ", self.n_class_labels)

        # stop recursively building the tree(split the node) when stopping criteria meets
        # assign the most common label at this moment to be the label of this node
        if self._is_finished(y, depth):
            u, counts = np.unique(y, return_counts=True)
            # if len(counts) == 0:
            #     print('=====')
            #     print(X)
            #     print('=====')
            #     print(y)
            #     print('=====')
            #     print(counts)
            #     print('=====')
            most_common_Label = u[np.argmax(counts)]
            return Node(label=most_common_Label)

        # randomize N features
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        # get best split feature and value out of N features
        best_split_feat, best_split_value = self._best_split(X, y, rnd_feats)

        # split on best split feature and value
        left_idx, right_idx = self._create_split(X[:, best_split_feat], best_split_value)
        # build DT recursively for each subset
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_split_feat, best_split_value, left_child, right_child)

    # check whether stopping criteria meets
    def _is_finished(self, y, depth):
        # if the tree is deep enough,
        # or only one class label left,
        # or sample is less than min_sample_split
        # or is pure enough, stop the split
        if (depth >= self.max_depth
                or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split
                or self._entropy(y) < self.impurity_stopping_threshold):
            return True
        return False

    # get best split feature and value out of N features
    def _best_split(self, X, y, rnd_features):
        split = {'IG': - 1, 'feat': None, 'value': None}

        # for each feature
        for feat in rnd_features:
            # print("------- Split on feature: ", feat, "-------")
            # get all values of one feature
            X_feat_column = X[:, feat]
            # get all split values of one feature
            split_values = np.unique(X_feat_column)
            # for each split value
            for value in split_values:
                # print("Value: ", value)
                # calculate IG if split on a particular value of a particular feature
                ig = self._information_gain(X_feat_column, y, value)
                # capture the highest information gain, split feature and split value
                if ig > split['IG']:
                    split['IG'] = ig
                    split['feat'] = feat
                    split['value'] = value
        # print("_b_s_ Best split feature: ", split['feat'],
        #       ", value: ", split['value'],
        #       ", IG: ", split['IG'], "\n")
        # return split feature and split value with the highest IG
        return split['feat'], split['value']

    # calculate information gain if split on a particular value of a particular feature
    def _information_gain(self, X_feat_column, y, split_val):
        # split on given feature and value
        left_idx, right_idx = self._create_split(X_feat_column, split_val)
        # count the 2 part after split
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)
        # if only one class left after split, rest IG
        if n_left == 0 or n_right == 0:
            # print("can't split into subset, return IG = 0")
            return 0

        # calculate entropy of 2 children
        child_entropy = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        # print("Parent entropy: ", self._entropy(y), ", child entropy: ", child_entropy, ", IG: ", self._entropy(y) - child_entropy)
        # print("Parent samples: ", n, ", left child samples: ", n_left, ", right child samples: ", n_right)
        # return IG = parent entropy - 2 children's entropy
        return self._entropy(y) - child_entropy

    # calculate entropy
    @staticmethod
    def _entropy(y):
        # get count of each label
        u, counts = np.unique(y, return_counts=True)
        # calculate proportions for each label
        proportions = counts / len(y)
        # calculate entropy
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    # split on given feature and value
    @staticmethod
    def _create_split(X_feat_column, split_val):
        left_idx = np.argwhere(X_feat_column <= split_val).flatten()
        right_idx = np.argwhere(X_feat_column > split_val).flatten()
        return left_idx, right_idx

    # prediction
    def predict(self, X: pd.DataFrame):
        # call the predict method
        X_array = X.to_numpy()
        return self._predict(X_array)

    def _predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.label

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# clean up raw data
def check_position(sent, posi):
    sent = str(sent).split(" ")
    index = -1
    if '_______' in sent:
        index = sent.index('_______')
    return posi == index


# get target around words list
def split_features(sent, posi, rang):
    sent = str(sent).split(" ")
    str_len = len(sent)
    list = [sent[int(posi) - i] if int(posi) - i >= 0 else "<BEGIN_SENTENCE>" for i in range(rang, 0, -1)] \
           + [sent[int(posi) + 1 + i] if int(posi) + 1 + i < str_len else "<END_SENTENCE>" for i in range(rang)]
    return list


# get one hot encoding dict
def onehot_map(vocabs):
    df = pd.get_dummies(pd.Series(vocabs))
    df['one_hot'] = df.apply(lambda x: ','.join(x.astype(str)), axis=1)
    r = dict(zip(vocabs, df.one_hot))
    return r


# create dataset
# def create_dataset(file, rang, direction='both', vocabs_oh=None):
#     # get file content into DF
#     df = pd.DataFrame(open(file, "r").readlines(), columns=['ORG_DATA'])
#
#     # split into 3 columns
#     df[['Target', 'Position', 'Sentence']] = df.ORG_DATA.apply(lambda x: pd.Series(str(x).split(" ", 2)))
#
#     # clean up
#     df['check'] = df.apply(lambda x: check_position(x['Sentence'], int(x['Position'])), axis=1)
#     df = df[df['check']]
#
#     before_after = df.apply(lambda x: split_features(x['Sentence'], x['Position'], rang), axis=1)
#
#     if direction == 'before':
#         # get words list around target for each sample
#         df['Words_List'] = before_after.apply(lambda x: x[:rang])
#     elif direction == 'after':
#         df['Words_List'] = before_after.apply(lambda x: x[rang:])
#     else:
#         df['Words_List'] = before_after
#
#     # create vocabulary using Train dataset. Re-use the vovabulary for Dev and Test dataset
#     if vocabs_oh is None:
#         # get vocabulary(unique words)
#         vocabs = df['Words_List'].explode().unique()
#         vocabs = np.append(vocabs, '<UNKNOWN_WORD>')
#         # one hot mapping based on vocabs
#         vocabs_oh = onehot_map(vocabs)
#
#     # replace each word with one hot mapping
#     X = pd.DataFrame(df.Words_List.tolist(), index=df.index)
#     #     X.columns = range(rang*2)
#     X = X.applymap(lambda i: vocabs_oh[i] if i in vocabs_oh else vocabs_oh['<UNKNOWN_WORD>'])
#
#     # Generate one hot mapped feature dataframe
#     temp = X.apply(lambda x: ','.join(x.astype(str)), axis=1)
#     Xs = temp.apply(lambda x: pd.Series(str(x).split(",")))
#
#     y = df['Target']
#
#     return Xs, y, vocabs_oh


def write_dict_to_file(dic, file_name):
    with open(file_name, 'w') as f: f.write(json.dumps(dic))


def _test():
    df = pd.read_csv('tennis_dataset/tennis.csv')
    X = df.drop(['Label'], axis=1)
    y = df['Label'].apply(lambda x: 0 if x == '+' else 1)

    df = pd.read_csv('tennis_dataset/tennis_test.csv')
    X_test = df.drop(['Label'], axis=1)
    y_test = df['Label'].apply(lambda x: 0 if x == '+' else 1)

    clf = DecisionTreeModel()
    clf.fit(X, y)

    y_pred = clf.predict(X_test)
    print(y_test.shape)
    print(y_pred.shape)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)


if __name__ == "__main__":
    _test()
