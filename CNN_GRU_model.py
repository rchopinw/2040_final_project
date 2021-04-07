import tensorflow as tf


class HyperParams:
    num_words = 92
    num_docs = 50
    sequence_len = 20
    batch_size = 8
    vocab_size = 252911
    embedded_size = 256
    num_units_bi_gru_1 = 128
    num_units_main_gru = 512
    num_units_main_gru_2 = 128
    num_units_post_dense_1 = 300
    num_units_post_dense_2 = 64
    num_output = 1
    auxiliary_x = 6

    lr = 1e-05
    opt = tf.keras.optimizers.Adam(lr)
    loss = tf.keras.losses.MeanSquaredError()
    epochs = 100
    metrics = ['mean_absolute_error', 'mean_squared_error', 'mean_squared_logarithmic_error']
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='min',
                                          patience=10)
    r = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.8,
                                             patience=2,
                                             verbose=1,
                                             mode='auto',
                                             epsilon=0.0001,
                                             cooldown=5,
                                             min_lr=0.00001)


def tri_attention_rnn_model(plot_model=True,
                            summary_model=True):
    word_input = tf.keras.Input(shape=HyperParams.num_words,
                                dtype='float32')

    # Embedding layer
    word_embedding = tf.keras.layers.Embedding(input_dim=HyperParams.vocab_size,
                                               output_dim=HyperParams.embedded_size)(word_input)
    bi_gru_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(HyperParams.num_units_bi_gru_1,
                                                                 return_sequences=True))(word_embedding)
    # the first attention layer to capture the word-level attentions
    attention_dense_1 = tf.keras.layers.Dense(HyperParams.num_units_bi_gru_1 * 2,
                                              activation='tanh')(bi_gru_1)
    attention_softmax_1 = tf.keras.layers.Activation('softmax')(attention_dense_1)
    attention_mul_1 = tf.keras.layers.multiply([attention_softmax_1, bi_gru_1])
    vec_sum_1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attention_mul_1)
    word_level_model = tf.keras.Model(inputs=word_input,
                                      outputs=vec_sum_1)

    doc_input = tf.keras.Input(shape=(HyperParams.num_docs, HyperParams.num_words),
                               dtype='float32')
    post_bi_gru_1 = tf.keras.layers.TimeDistributed(word_level_model)(doc_input)

    # the second attention layer to capture the news-level attentions
    attention_dense_2 = tf.keras.layers.Dense(HyperParams.num_units_bi_gru_1 * 2,
                                              activation='tanh')(post_bi_gru_1)
    attention_softmax_2 = tf.keras.layers.Activation('softmax')(attention_dense_2)
    attention_mul_2 = tf.keras.layers.multiply([attention_softmax_2, post_bi_gru_1])
    vec_sum_2 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attention_mul_2)
    doc_level_model = tf.keras.Model(inputs=doc_input, outputs=vec_sum_2)

    seq_input = tf.keras.Input(shape=(HyperParams.sequence_len, HyperParams.num_docs, HyperParams.num_words),
                               dtype='float32')

    # Incorporating the auxiliary variables
    auxiliary_input = tf.keras.Input(shape=(HyperParams.sequence_len, HyperParams.auxiliary_x),
                                     dtype='float32')
    pre_main_gru = tf.keras.layers.TimeDistributed(doc_level_model)(seq_input)
    pre_main_gru_concat = tf.keras.layers.concatenate([pre_main_gru, auxiliary_input], axis=-1)
    main_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(HyperParams.num_units_main_gru,
                                                                 return_sequences=True))(pre_main_gru_concat)
    main_gru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(HyperParams.num_units_main_gru_2,
                                                                   return_sequences=True))(main_gru)

    # the third attention layer to capture the time-level attentions
    attention_dense_3 = tf.keras.layers.Dense(HyperParams.num_units_main_gru_2 * 2,
                                              activation='tanh')(main_gru_2)
    attention_softmax_3 = tf.keras.layers.Activation('softmax')(attention_dense_3)
    attention_mul_3 = tf.keras.layers.multiply([attention_softmax_3, main_gru_2])
    vec_sum_3 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attention_mul_3)

    # Passing through multiple FC layers
    main_dense_1 = tf.keras.layers.Dense(HyperParams.num_units_post_dense_1,
                                         activation='tanh',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001,
                                                                                        l2=0.00001))(vec_sum_3)
    main_dropout_1 = tf.keras.layers.Dropout(0.1)(main_dense_1)
    main_dense_2 = tf.keras.layers.Dense(HyperParams.num_units_post_dense_2,
                                         activation='tanh',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001,
                                                                                        l2=0.00001))(main_dropout_1)
    main_dropout_2 = tf.keras.layers.Dropout(0.1)(main_dense_2)
    output = tf.keras.layers.Dense(HyperParams.num_output,
                                   activation=None)(main_dropout_2)

    model = tf.keras.Model(inputs=[seq_input, auxiliary_input], outputs=output)

    if plot_model:
        tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
    if summary_model:
        print(model.summary())
    return model


def compile_model(m):
    m.compile(optimizer=HyperParams.opt,
              loss=HyperParams.loss,
              metrics=HyperParams.metrics)
    return m


def parse_data(x):
    description = {'x1': tf.io.FixedLenFeature([], tf.string, default_value=''),
                   'x2': tf.io.FixedLenFeature([], tf.string, default_value=''),
                   'y': tf.io.FixedLenFeature([], tf.float32, default_value=-1)}
    parsed_features = tf.io.parse_single_example(x, description)
    x1 = tf.reshape(parsed_features['x1'], [HyperParams.sequence_len, HyperParams.num_docs, HyperParams.num_words])
    x1 = tf.cast(x1, tf.float32)

    x2 = tf.reshape(parsed_features['x2'], [HyperParams.sequence_len, HyperParams.auxiliary_x])
    x2 = tf.cast(x2, tf.float32)

    y = parsed_features['y']

    return x1, x2, y


def load_data(file):
    df = tf.data.TFRecordDataset(file)
    df = df.map(parse_data)
    return df


if __name__ == '__main__':
    m1 = tri_attention_rnn_model()
    m1 = compile_model(m1)
    history = m1.fit()
