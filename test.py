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
