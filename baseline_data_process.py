import numpy as np
import json
import pandas as pd
import os
import random
import datetime
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from functools import reduce
import gensim

data_files = ['azn', 'biontech', 'jnj', 'moderna', 'novavax', 'pfizer']


def save_obj(obj, name):
    """
    :param obj:
    :param name:
    :return:
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    :param name:
    :return:
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def list_major_and_related(dir):
    """
    :param dir:
    :return:
    """
    files = os.listdir(dir)
    major_tweets_f = []
    related_tweets_f = []
    for fn in files:
        if 'related' in fn:
            related_tweets_f.append(fn)
        else:
            major_tweets_f.append(fn)
    return major_tweets_f, related_tweets_f


def read_file(n):
    """
    :param n:
    :return:
    """
    with open(n, 'rb') as f:
        fr = f.readlines()

    def decoder(s):
        try:
            ds = str(s, 'utf_8_sig')
        except UnicodeDecodeError:
            try:
                ds = str(s, 'utf-8')
            except UnicodeDecodeError:
                try:
                    ds = str(s, 'gbk')
                except UnicodeDecodeError:
                    try:
                        ds = str(s, 'GB18030')
                    except UnicodeDecodeError:
                        ds = str(s, 'gb2312', 'ignore')
        return ds

    fr = list(map(lambda x: json.loads(decoder(x)), fr))
    return fr


def process_file(file_dir: str,
                 company_name: str,
                 abandon: list = None) -> pd.DataFrame:
    """
    Processing the related twitters regarding company_name
    :param file_dir: file directory
    :param company_name: target company name
    :param abandon: the columns to abandon
    :return: a pandas dataframe of cleaned data
    """
    if abandon is None:
        abandon = ['conversation_id', 'created_at', 'time', 'timezone', 'user_id',
                   'username', 'name', 'place', 'mentions', 'link', 'quote_url',
                   'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
                   'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src', 'trans_dest']
    op = read_file(file_dir)
    d = pd.DataFrame(op)
    d = d.drop_duplicates(subset=['id'])
    d = d.drop(columns=abandon)
    d = d[d['language'] == 'en'].reset_index(drop=True)
    d = d.drop(columns=['language'])
    # process date
    d['date'] = d['date'].apply(lambda x: x.replace('-', ''))
    # process urls:
    d['linked_url'] = d['urls'].apply(lambda x: 1 if x else 0)
    d = d.drop(columns=['urls'])
    # process photos:
    d['with_photo'] = d['photos'].apply(lambda x: 1 if x else 0)
    d = d.drop(columns=['photos'])
    # process retweet
    d['retweet'] = d['retweet'].apply(lambda x: 1 if x else 0)
    # process video
    d['video'] = d['video'].apply(lambda x: 1 if x else 0)
    # process date:
    d['date'] = d['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    d['date'] = d['date'].apply(
        lambda x: x + datetime.timedelta(days=-1) if x.weekday() == 5 else x + datetime.timedelta(
            days=-2) if x.weekday() == 6 else x)
    d['date'] = d['date'].apply(lambda x: datetime.datetime.strftime(x, '%Y%m%d'))
    # process format:
    d['replies_count'] = d['replies_count'].astype('int64')
    d['retweets_count'] = d['retweets_count'].astype('int64')
    d['likes_count'] = d['likes_count'].astype('int64')
    d['retweet'] = d['retweet'].astype('int64')
    d['video'] = d['video'].astype('int64')
    d['linked_url'] = d['linked_url'].astype('int64')
    d['with_photo'] = d['with_photo'].astype('int64')
    # group by date
    new_df = [dict(zip(sub_df.columns, [sum(sub_df[x])
                                        if sub_df[x].dtype == 'int64'
                                        else sub_df[x].tolist()
                                        for x in sub_df.columns]))
              for _, sub_df in d.groupby('date')]
    new_df = pd.DataFrame(new_df)
    new_df['company_name'] = company_name
    new_df['date'] = new_df['date'].apply(lambda x: x[0])
    return new_df


# process json file for each company
def gather_all(dfs: list = None) -> pd.DataFrame:
    """
    :param dfs:
    :return:
    """
    if dfs is None:
        dfs = data_files
    corp_gathered = []
    for f in dfs:
        mtf, rtf = list_major_and_related('./{}'.format(f))
        print('...Processing file {}...{}'.format(f, rtf))
        corp_file = []
        for j in rtf + mtf:
            j_d = './{}/'.format(f) + j
            corp_file.append(process_file(file_dir=j_d,
                                          company_name=f))
        corp_file = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), corp_file)
        corp_gathered.append(corp_file)
    corp_gathered = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), corp_gathered)
    corp_gathered = corp_gathered.drop_duplicates(subset=['date', 'company_name'], keep='first')
    return corp_gathered


def cum_sum(target):
    """
    :param target:
    :return:
    """
    s = sum(target)
    r = list(range(s))
    res = []
    for i in target:
        tmp = r[0:i]
        r = r[i:]
        res.append(tmp)
    return res


def df_process(df,
               drop_columns=None,
               norm=True,
               norm_columns=None,
               norm_approach='min-max'):
    """
    :param df:
    :param drop_columns:
    :param norm:
    :param norm_columns:
    :param norm_approach:
    :param predict_time:
    :return:
    """
    if drop_columns is None:
        drop_columns = ['open', 'high', 'low', 'close',
                        'adjclose', 'volume', 'company_name', 'date',
                        'hashtags', 'cashtags', 'retweet']
    df = df.drop(columns=drop_columns)
    df = df.dropna()
    if norm:
        if norm_columns is None:
            norm_columns = ['replies_count', 'retweets_count',
                            'likes_count',
                            'video', 'linked_url', 'with_photo', 'target_variation']
        if norm_approach == 'min-max':
            s = MinMaxScaler()
        else:
            s = StandardScaler()
        s.fit(df[norm_columns])
        s_f = np.array(s.transform(df[norm_columns]))
    else:
        s_f = np.array(df)
    y = s_f[:, -1]
    x = s_f[:, :-1]
    return x, y


# generate data for baseline model
class ProcessR2(object):
    def __init__(self,
                 text,
                 vs=100,
                 ws=5,
                 mc=2,
                 wks=4):
        self.text = text
        print('...{} Documents Loaded...'.format(len(self.text)))
        self.docs = self.preprocess_docs()
        self.tagged_docs = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(self.docs)]
        self.model = self.d2v(vs, ws, mc, wks)
        self.doc_vs = self.doc_v()
        print('...Language Processing and Language Model Training Finished...')

    def preprocess_docs(self):
        # Including steps: removing punctuations,
        #                  removing duplicated white spaces,
        #                  remove numbers,
        #                  remove stopwords,
        #                  stemming the text.
        return gensim.parsing.preprocessing.preprocess_documents(self.text)

    def d2v(self, vs, ws, mc, wks):
        # reference: https://arxiv.org/pdf/1405.4053v2.pdf <Distributed Representations of Sentences and Documents>
        print('...Training Doc2Vector Model via Gensim.models.Doc2Vec...')
        model = gensim.models.doc2vec.Doc2Vec(self.tagged_docs,
                                              vector_size=vs,
                                              window=ws,
                                              min_count=mc,
                                              workers=wks,
                                              compute_loss=True)
        return model

    def doc_v(self):
        return [self.model[x[1][0]].tolist() for x in self.tagged_docs]

    def process_text(self, s):
        return gensim.parsing.preprocessing.preprocess_string(s)

    def infer(self, dv):
        return self.model.infer_vector(dv)


def generate_sequence_data(pr, map_relation, d, sequence_len=20):
    text_vecs = np.zeros((d.shape[0], len(pr.doc_vs[0])))
    for idx, content in enumerate(zip(d.date, d.company_name)):
        date, company_name = content
        key = "{}_{}".format(date, company_name)
        s = np.array([pr.doc_vs[x] for x in map_relation[key]])
        text_vecs[idx, :] = np.mean(s, axis=0)
    x, y = df_process(d)
    features = pd.DataFrame(np.concatenate([text_vecs, x], axis=1))
    features['date'] = d.date
    features['y'] = y
    features['company_name'] = d.company_name
    xs = []
    ys = []
    for company_name, sub_df in features.groupby('company_name'):
        tmp = sub_df.sort_values(by=['date'], ascending=True)
        for idx in range(sub_df.shape[0] - sequence_len):
            sub_x = sub_df.iloc[idx:idx + sequence_len, :].drop(columns=['date', 'company_name', 'y'])
            sub_y = sub_df.iloc[idx + sequence_len, :]['y'].tolist()
            xs.append((np.array(sub_x)).tolist())
            ys.append(sub_y)
    return xs, ys


related_corp_tweets = pd.read_csv('/content/drive/MyDrive/2040 final/2040 final proj/data_op.csv',
                                  encoding='utf_8_sig')
stock_price = pd.read_csv('/content/drive/MyDrive/2040 final/2040 final proj/stockprice.csv', encoding='utf_8_sig',
                          index_col=0)

related_corp_tweets['date'] = related_corp_tweets['date'].astype('str')
stock_price['date'] = stock_price['date'].astype('str')
# mapping date to tweet
tweet_stock_merge = pd.merge(related_corp_tweets, stock_price, on=['company_name', 'date'], how='right')
tweet_stock_merge = tweet_stock_merge.dropna().reset_index(drop=True)
date_text_related = dict(zip(tweet_stock_merge.date + '_' + tweet_stock_merge.company_name,
                             cum_sum([len(eval(x)) for x in tweet_stock_merge.tweet.tolist()])))

# format the texts
text_related = sum([eval(x) for x in tweet_stock_merge.tweet.tolist()], [])

related_pr2 = ProcessR2(text_related)

x_data, y_data = generate_sequence_data(pr=related_pr2,
                                        map_relation=date_text_related,
                                        d=tweet_stock_merge)
x_train, x_test, y_train, y_test = train_test_split(np.array(x_data), np.array(y_data), test_size=0.2, shuffle=True)

save_obj({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, 'baseline_model_data')