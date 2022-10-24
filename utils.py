import json
import pandas as pd
import numpy as np


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
def create_dataset(file, rang, direction='both', vocabs_oh=None, frac=1.0):
    # get file content into DF
    df = pd.DataFrame(open(file, "r").readlines(), columns=['ORG_DATA'])
    df = df.sample(frac=frac)

    # split into 3 columns
    df[['Target', 'Position', 'Sentence']] = df.ORG_DATA.apply(lambda x: pd.Series(str(x).split(" ", 2)))

    # clean up
    df['check'] = df.apply(lambda x: check_position(x['Sentence'], int(x['Position'])), axis=1)
    df = df[df['check']]

    before_after = df.apply(lambda x: split_features(x['Sentence'], x['Position'], rang), axis=1)
    # print(before_after)
    if direction == 'before':
        # get words list around target for each sample
        df['Words_List'] = before_after.apply(lambda x: x[:rang])
    elif direction == 'after':
        df['Words_List'] = before_after.apply(lambda x: x[rang:])
    else:
        df['Words_List'] = before_after

    # print(df)

    df['words_str'] = df.apply(lambda x: ' '.join(x['Words_List']), axis=1)

    # remove duplicate rows:
    # rows with same features and same label - remove all but 1
    # rows with same features but different labels are all removed
    df['duplicated_with_target'] = df.duplicated(subset=['words_str', 'Target'])
    df = df[~df['duplicated_with_target']]
    df['duplicated_features'] = df.duplicated(subset=['words_str'], keep=False)
    df = df[~df['duplicated_features']]

    # print(df)
    # create vocabulary using Train dataset. Re-use the vovabulary for Dev and Test dataset
    if vocabs_oh is None:
        # get vocabulary(unique words)
        vocabs = df['Words_List'].explode().unique()
        vocabs = np.append(vocabs, '<UNKNOWN_WORD>')
        # one hot mapping based on vocabs
        vocabs_oh = onehot_map(vocabs)

    # replace each word with one hot mapping
    X = pd.DataFrame(df.Words_List.tolist(), index=df.index)

    y = df['Target']

    X = X.applymap(lambda i: vocabs_oh[i] if i in vocabs_oh else vocabs_oh['<UNKNOWN_WORD>'])

    # Generate one hot mapped feature dataframe
    temp = X.apply(lambda x: ','.join(x.astype(str)), axis=1)
    Xs = temp.apply(lambda x: pd.Series(str(x).split(",")))

    return Xs, y, vocabs_oh


def write_dict_to_file(dic, file_name):
    with open(file_name, 'w') as f: f.write(json.dumps(dic))
