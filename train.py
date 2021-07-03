from DAR import baseline_model_text

x_train, y_train, x_test, y_test, word_index = text_prepare_newWithPadding()
baseline_model_text(x_train, y_train, x_test, y_test, word_index)
