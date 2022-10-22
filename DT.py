import pandas as pd
import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold  # remove?
        self.left = left  # why left, right?
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
        print("********** Depth # ***********", depth,
              ", # samples: ", self.n_samples,
              ", # features: ", self.n_features,
              ", # labels: ", self.n_class_labels)

        # stop recursively building the tree(split the node) when stopping criteria meets
        # assign the most common label at this moment to be the label of this node
        if self._is_finished(y, depth):
            u, counts = np.unique(y, return_counts=True)
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
        # return ????
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
            print("------- Split on feature: ", feat, "-------")
            # get all values of one feature
            X_feat_column = X[:, feat]
            # get all split values of one feature
            split_values = np.unique(X_feat_column)
            # for each split value
            for value in split_values:
                print("Value: ", value)
                # calculate IG if split on a particular value of a particular feature
                ig = self._information_gain(X_feat_column, y, value)
                # capture the highest information gain, split feature and split value
                if ig > split['IG']:
                    split['IG'] = ig
                    split['feat'] = feat
                    split['value'] = value
        print("_b_s_ Best split feature: ", split['feat'],
              ", value: ", split['value'],
              ", IG: ", split['IG'], "\n")
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
            print("can't split into subset, return IG = 0")
            return 0

        # calculate entropy of 2 children
        child_entropy = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        print("Parent entropy: ", self._entropy(y), ", child entropy: ", child_entropy, ", IG: ", self._entropy(y) - child_entropy)
        print("Parent samples: ", n, ", left child samples: ", n_left, ", right child samples: ", n_right)
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


def _test():
    df = pd.read_csv('tennis.csv')
    X = df.drop(['Label'], axis=1)
    y = df['Label'].apply(lambda x: 0 if x == '+' else 1)

    df = pd.read_csv('tennis_test.csv')
    X_test = df.drop(['Label'], axis=1)
    y_test = df['Label'].apply(lambda x: 0 if x == '+' else 1)

    clf = DecisionTreeModel()
    clf.fit(X, y)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    # x = "whether 4 it was not clear _______ the militia had arrested the women or why ."
    # y = x.split(" ", 2)
    # print(len(y))
    # print(y[0])
    # print(y[1])
    # print(y[2])


def _split_features(sentence, position):
    after = str(sentence).split(" ")
    return [after[int(position) - 1], after[int(position) + 1]]


if __name__ == "__main__":
    _test()

    # file = open("hw1.train.col", "r")
    # org_train = file.readlines()
