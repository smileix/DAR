# swda数据处理
from swda import Transcript
trans = Transcript('swda/sw00utt/sw_0001_4325.utt.csv', 
    'swda/swda-metadata.csv')

# 转录对象生成
utt = trans.utterances[19]
utt.pos_words()
['I', 'guess', '--']
utt.act_tag
'sv'

# 模型搭建
input = Input(shape=(MAX_CONVERSATION_LENGTH, MAX_SEQUENCE_LENGTH))
xl = Lambda(lambda x: K.reshape(x, [-1, MAX_SEQUENCE_LENGTH]))(input)
xl = Embedding(len(word_index) + 1, EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=True, mask_zero=True)(xl)
xl = Bidirectional(LSTM(hidden_layer,dropout=0.5,
    return_sequences=True,
    kernel_initializer='random_uniform',
    recurrent_initializer='glorot_uniform'))(xl)
xl = TimeDistributed(Dense(hidden_layer, 
    input_shape=(MAX_SEQUENCE_LENGTH, hidden_layer)))(xl)
xl = AttentionwithContext()(xl)
xl = Lambda(lambda x: K.reshape(x, 
    [-1,MAX_CONVERSATION_LENGTH, hidden_num]))(xl)
xl = Bidirectional(LSTM(hidden_layer, 
    input_shape = (MAX_CONVERSATION_LENGTH, hidden_layer),
    return_sequences=True,
    kernel_initializer='random_uniform',
    recurrent_initializer='glorot_uniform',))(xl)
xl = TimeDistributed(Dense(output_layer, 
    input_shape=(MAX_CONVERSATION_LENGTH, hidden_layer)))(xl)
crf = CRF(True) # 定义crf层，参数为True，自动mask掉最后一个标签
tag_score = Dense(output_num)(xl)
tag_score = crf(tag_score) # 包装一下原来的tag_score
optimizer = RMSprop(lr=learning_rate, decay=0.001)
model = Model(inputs=input, outputs=tag_score)
model.summary()
model.compile(loss=crf.loss,
    optimizer=optimizer,
    metrics=[crf.accuracy])
model.fit(x_train, y_train, 
    validation_data=(x_test, y_test), 
    nb_epoch=15, batch_size=64, verbose=2)
model.save('model_new/basemodel_new.h5')

# 模型预测
# 混淆矩阵
def cm_plot(y_test_label, predict_label):
    cm = confusion_matrix(y_test_label, predict_label)
    plt.matshow(cm, cmap=plt.cm.Blues) # 画混淆矩阵，配色风格使用
    plt.colorbar() # 颜色标签
    for x in range(42):
        for y in range(42):
            plt.annotate(cm[x,y], xy=(y,x), 
horizontalalignment='center', 
verticalalignment='center')
            plt.ylabel('True label') # 坐标轴标签
            plt.xlabel('Predicted label') # 坐标轴标签
    return plt

# 准确率
# 训练过程中显示逐帧准确率，排除了mask的影响
def accuracy(y_true, y_pred, ignore_last_label, num_labels): 
    y.true = y.true.astype(np.float64)
    mask = 1 - y_true[:,:,-1] if ignore_last_label else None
    y_true, y_pred = y_true[:,:,:num_labels], y_pred[:,:,:num_labels]
    isequal = np.equal(np.argmax(y_true, 2), np.argmax(y_pred, 2))
    isequal = isequal.astype(np.float64)
    return np.sum(isequal*mask) / np.sum(mask)

# 模型预测
y.predicted = model.predict(x_test)
accuracy_rate = accuracy(y_test, y_predicted, True, output_num)
print("acc", accuracy_rate)
y.predicted = np.reshape(y_predicted, [-1, output_num])
y.test = np.reshape(y_test, [-1, output_num])
predict_label = np.argmax(y_predicted, axis=1)
y_test_label = np.argmax(y_test, axis=1)
cm_plot(y_test_label, predict_label).show()
print(confusion_matrix(y_test_label, predict_label))
return accuracy_rate
