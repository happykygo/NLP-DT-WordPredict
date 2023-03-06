import json
import pandas as pd
import numpy as np


def check_position(sent, posi):
    """
    clean up raw data, check eligibility of given position for target word
    :param sent:
    :param posi: given index of target word
    :return:
    """
    sent = str(sent).split(" ")
    index = -1
    if '_______' in sent:
        index = sent.index('_______')
    return posi == index


def split_features(sent, posi, rang):
    """
    Get words list around target word within specified range. If the range exceeds sentence valid index, mark as
    <BEGIN_SENTENCE> or <END_SENTENCE>
    :param sent:
    :param posi: given index of target word
    :param rang: specified range around target word
    :return:
    """
    sent = str(sent).split(" ")
    str_len = len(sent)
    word_list = [sent[int(posi) - i] if int(posi) - i >= 0 else "<BEGIN_SENTENCE>" for i in range(rang, 0, -1)] + \
                [sent[int(posi) + 1 + i] if int(posi) + 1 + i < str_len else "<END_SENTENCE>" for i in range(rang)]
    return word_list


def onehot_map(vocabs):
    """
    get one hot encoding dictionary
    :param vocabs:
    :return:
    """
    df = pd.get_dummies(pd.Series(vocabs))
    df['one_hot'] = df.apply(lambda x: ','.join(x.astype(str)), axis=1)
    r = dict(zip(vocabs, df.one_hot))
    return r


# create dataset
def create_dataset(file, rang, direction='both', vocabs_oh=None, frac=1.0):
    # get file content into DF
    df = pd.DataFrame(open(file, "r").readlines(), columns=['ORG_DATA'])
    df = df.sample(frac=frac)

    # split into 3 columns
    df[['Target', 'Position', 'Sentence']] = df.ORG_DATA.apply(lambda x: pd.Series(str(x).split(" ", 2)))

    # clean up -- remove samples with invalid target word index
    df['check'] = df.apply(lambda x: check_position(x['Sentence'], int(x['Position'])), axis=1)
    df = df[df['check']]

    # Get words list around target word within specified range.
    before_after = df.apply(lambda x: split_features(x['Sentence'], x['Position'], rang), axis=1)
    if direction == 'before':
        df['Words_List'] = before_after.apply(lambda x: x[:rang])
    elif direction == 'after':
        df['Words_List'] = before_after.apply(lambda x: x[rang:])
    else:
        df['Words_List'] = before_after

    df['words_str'] = df.apply(lambda x: ' '.join(x['Words_List']), axis=1)

    # rows with same features and same label - keep first occurrence
    df['duplicated_with_target'] = df.duplicated(subset=['words_str', 'Target'])
    df = df[~df['duplicated_with_target']]
    # rows with same features but different labels - all removed
    df['duplicated_features'] = df.duplicated(subset=['words_str'], keep=False)
    df = df[~df['duplicated_features']]

    # create vocabulary of words around target within range using Train dataset.
    # Construct one hot mapping for vocabulary, which will be re-used for Dev and Test dataset
    if vocabs_oh is None:
        # get vocabulary(unique words)
        vocabs = df['Words_List'].explode().unique()
        vocabs = np.append(vocabs, '<UNKNOWN_WORD>')
        # one hot mapping based on vocabs
        vocabs_oh = onehot_map(vocabs)

    # replace each word with one hot mapping
    X = pd.DataFrame(df.Words_List.tolist(), index=df.index)
    X = X.applymap(lambda i: vocabs_oh[i] if i in vocabs_oh else vocabs_oh['<UNKNOWN_WORD>'])

    # Generate one hot mapped feature dataframe
    temp = X.apply(lambda x: ','.join(x.astype(str)), axis=1)
    Xs = temp.apply(lambda x: pd.Series(str(x).split(",")))

    y = df['Target']

    return Xs, y, vocabs_oh


def write_dict_to_file(dic, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(dic))
