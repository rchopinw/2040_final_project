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


class ProcessR(object):
    def __init__(self,
                 text,
                 filters='',
                 padding='post'):
        """
        :param text:
        :param filters:
        :param padding:
        """
        self.text = text
        self.padding = padding
        self.filters = filters
        print('...{} Documents Loaded...'.format(len(self.text)))
        self.docs = self._preprocess_docs()
        print('...Language Preprocess Finished...')
        self.docs_f, self.tokenizer_f = self._tokenize()
        print('...Language Tokenizing Finished...')

    def _preprocess_docs(self):
        # Including steps: removing punctuations,
        #                  removing duplicated white spaces,
        #                  remove numbers,
        #                  remove stopwords,
        #                  stemming the text.
        return gensim.parsing.preprocessing.preprocess_documents(self.text)

    def _tokenize(self):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=self.filters)
        lang_tokenizer.fit_on_texts(self.docs)
        tensor = lang_tokenizer.texts_to_sequences(self.docs)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding=self.padding)
        return tensor.tolist(), lang_tokenizer


def generate_tf_rec(processor: ProcessR,
                    f_m: pd.DataFrame,
                    sequence_length: int,
                    map_relation: dict,
                    num_per_day: int,
                    n_repeat: int,
                    test_size: float):
    """
    :param f_m:
    :param test_size:
    :param n_repeat:
    :param num_per_day:
    :param processor:
    :param map_relation:
    :param sequence_length:
    :return:
    """
    x, y = df_process(f_m)
    f_m['target_variation'] = y
    train_writer = tf.io.TFRecordWriter(path='train.tfrecords')
    validation_writer = tf.io.TFRecordWriter(path='test.tfrecords')
    for date, company in zip(f_m.date, f_m.company_name):
        print('Executing {}-{}'.format(date, company))
        sub_df = f_m[f_m['company_name'] == company]
        sub_df = sub_df.sort_values(by=['date'], ascending=True)
        sub_x = x[sub_df.index, :]  # keep the sequential pattern
        sub_df = sub_df.reset_index(drop=True)  # reset the index to correctly locate the sequence index
        ext_date = sub_df[sub_df['date'] == date]
        if ext_date.shape[0] > 1:
            print('Warning, multiple rows detected while allocating via [date, company].')
        try:
            fluctuation_norm = sub_df.iloc[ext_date.index[0] + sequence_length, :]['target_variation']
        except IndexError:
            continue
        sequence_df = sub_df.iloc[ext_date.index[0]:ext_date.index[0] + sequence_length, :]
        sequence_x = sub_x[ext_date.index[0]:ext_date.index[0] + sequence_length, :]
        for _ in range(random.randint(n_repeat - 10, n_repeat + 10)):
            sub_rec = []
            for k in [x + '_{}'.format(company) for x in sequence_df['date']]:
                tmp = [processor.docs_f[x] for x in map_relation[k]]
                tmp = [tmp[x] for x in [random.randint(0, len(tmp) - 1) for _ in range(num_per_day)]]
                sub_rec.append(tmp)
            print(sub_rec)
            sub_rec = np.array(sub_rec)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'x1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sub_rec.tostring()])),
                    'x2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sequence_x.tostring()])),
                    'y': tf.train.Feature(float_list=tf.train.FloatList(value=[fluctuation_norm]))
                }
            ))
            if random.random() < test_size:
                validation_writer.write(example.SerializeToString())
            else:
                train_writer.write(example.SerializeToString())
    validation_writer.close()
    train_writer.close()
    print('Finished.')


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


if __name__ == '__main__':
    gather_all().to_csv('data_op.csv', encoding='utf_8_sig', index=False)
    related_corp_tweets = pd.read_csv('data_op.csv', encoding='utf_8_sig')
    stock_price = pd.read_csv('stockprice.csv', encoding='utf_8_sig', index_col=0)

    related_corp_tweets['date'] = related_corp_tweets['date'].astype('str')
    stock_price['date'] = stock_price['date'].astype('str')
    # mapping date to tweet
    tweet_stock_merge = pd.merge(related_corp_tweets, stock_price, on=['company_name', 'date'], how='right')
    tweet_stock_merge = tweet_stock_merge.dropna().reset_index(drop=True)
    date_text_related = dict(zip(tweet_stock_merge.date + '_' + tweet_stock_merge.company_name,
                                 cum_sum([len(eval(x)) for x in tweet_stock_merge.tweet.tolist()])))

    # format the texts
    text_related = sum([eval(x) for x in tweet_stock_merge.tweet.tolist()], [])

    related_pr = ProcessR(text_related)

    generate_tf_rec(processor=related_pr,
                    f_m=tweet_stock_merge,
                    map_relation=date_text_related,
                    sequence_length=20,
                    num_per_day=50,
                    n_repeat=20,
                    test_size=0.2)
