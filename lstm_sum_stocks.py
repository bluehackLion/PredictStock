#!/usr/bin/env python
import os,sys
import random
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('input_data_column_cnt', '7', 'input num of data columnes')
flags.DEFINE_integer('output_data_column_cnt', '1', 'output num of data columnes')
flags.DEFINE_integer('seq_length', '12', 'length of sequence')
flags.DEFINE_integer('rnn_cell_hidden_dim', '30', 'num of LSTM hidden dimention')
flags.DEFINE_integer('num_stacked_layers', '12', 'num of stack LSTM layers')
flags.DEFINE_integer('epoch_num', '10000', 'num of epoch')
flags.DEFINE_float('forget_bias', '1.0', 'bias of forget gate')
flags.DEFINE_float('keep_prob', '1.0', 'when dropout, keep rate')
flags.DEFINE_float('learning_rate', '0.01', 'learning rate')


#정규화 min-max scaling
def min_max_scaling(x):
  x_np = np.asarray(x)
  return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

#정규화 복원
def reverse_min_max_scaling(org_x, x):
  org_x_np = np.asarray(org_x)
  x_np = np.asarray(x)
  return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

# train stock data loading
def create_dataset(stock_path, train_flag, seq_length):
  stock_list = os.listdir(stock_path)
  print(stock_list)
  encoding = 'euc-kr'
  names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
  X = []  # input dataset
  Y = []  # output dataset
  dataX = []  # sequence dataset
  dataY = []  # sequence dataset
  for idx in range(len(stock_list)):
    stock_file_path = os.path.join(stock_path ,stock_list[idx])
    raw_dataframe = pd.read_csv(stock_file_path, names=names, encoding=encoding)
   
    # delete 'Date' column
    del raw_dataframe['Date']
    stock_info = raw_dataframe.values[1:].astype(np.float)
    
    # price normalization
    open_price = stock_info[:,0:1]
    high_price = stock_info[:,1:2]
    low_price = stock_info[:,2:3]
    close_price = stock_info[:,3:4]
    prices = np.concatenate((open_price, close_price, high_price, low_price), axis=1)
    norm_price = min_max_scaling(prices)    # 정규화

    # volumes normalization
    volume = stock_info[:,-1:]
    norm_volume = min_max_scaling(volume)
                                  
    # high - close, low - close normalization, close - open
    sub_high_close = high_price - close_price
    sub_low_close = low_price - close_price
    sub_close_open = close_price - open_price
    norm_sub_high = min_max_scaling(sub_high_close)
    norm_sub_low = min_max_scaling(sub_low_close)
    norm_sub_open = min_max_scaling(sub_close_open)
    norm_sub_price = np.concatenate((norm_sub_high, norm_sub_low, norm_sub_open), axis=1)

    # price norm, volumes norm, sub norm 합침
    x = np.concatenate((norm_price, norm_sub_price,norm_volume), axis=1)
    # select target == Close price
    y = x[:,1:2]

    # seq_lenght == 12 시계열 데이터 입력 개수(ex 12일이 한번에 입력값으로 들어감)
    for i in range(0, len(y) - seq_length):
      _x = x[i : i + seq_length]
      _y = y[i + seq_length]
      dataX.append(_x)
      dataY.append(_y)
        
  # data shuffle 
  matrix = [[0 for col in range(2)] for row in range(len(dataY))]
  for i in range(len(dataY)):
    matrix[i][0] = dataX[i]
    matrix[i][1] = dataY[i]

  if train_flag:
    random.shuffle(matrix)

  for i in range(len(dataY)):
    X.append(matrix[i][0])
    Y.append(matrix[i][1])

  return (X, Y, prices)

# LSTM
def lstm_cell(rnn_cell_hidden_dim, forget_bias, keep_prob):
  cell = tf.contrib.rnn.BasicLSTMCell(
          num_units=rnn_cell_hidden_dim, 
          forget_bias=forget_bias, 
          state_is_tuple=True, 
          activation=tf.nn.softsign)
  
  if keep_prob < 1.0:
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
  return cell


def main(argv=None):
  tf.set_random_seed(777)
  rnn_cell_hidden_dim = FLAGS.rnn_cell_hidden_dim
  forget_bias = FLAGS.forget_bias
  keep_prob = FLAGS.keep_prob

  #make train dataset, val dataset
  train_data_path = 'stocks_train/' 
  val_data_path = 'stocks_val/'

  (train_dataX, train_dataY, test_prices) = create_dataset(train_data_path, True, FLAGS.seq_length)
  (val_dataX, val_dataY, val_prices) = create_dataset(val_data_path, False, FLAGS.seq_length)


  trainX = np.array(train_dataX)
  trainY = np.array(train_dataY)
  testX = np.array(val_dataX)
  testY = np.array(val_dataY)

  # make tensorflow placeholder 
  # X == input, Y == output
  X = tf.placeholder(tf.float32, [None, FLAGS.seq_length, FLAGS.input_data_column_cnt])  # (?, 12, 6)
  Y = tf.placeholder(tf.float32, [None, 1])            # (?, 1)
  targets = tf.placeholder(tf.float32, [None, 1])      # (?, 1)
  predictions = tf.placeholder(tf.float32, [None, 1])  # (?, 1)

  # num_stacked_layers개의 층으로 쌓인 stackedRNN 생성
  stackedRNNs = [lstm_cell(rnn_cell_hidden_dim, forget_bias, keep_prob) for _ in range(FLAGS.num_stacked_layers)]
  multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if FLAGS.num_stacked_layers > 1 else lstm_cell(rnn_cell_hidden_dim, forget_bias, keep_prob)

  (hypothesis, _states) = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

  hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], FLAGS.output_data_column_cnt, activation_fn=tf.identity)

  # loss func = 평균제곱오차, Optimizer func = AdamOptimizer
  loss = tf.reduce_sum(tf.square(hypothesis - Y))
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  train = optimizer.minimize(loss)

  # RMSE(Root Mean Square Error)
  rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

  train_error_summary = []  # 학습 데이터의 오류 기록
  test_error_summary = []   # 테스트 데이터의 오류 기록
  test_predict = ''         # 데스트 결과

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # train
  print("Start training!")
  for epoch in range(FLAGS.epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch+1)%500 == 0) or (epoch == FLAGS.epoch_num-1):  # 100번째 와 마지막일때 loss 출력
      # 학습데이터의 rmse오차 
      train_predict = sess.run(hypothesis, feed_dict={X: trainX})
      train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
      train_error_summary.append(train_error)
        
      # 테스트 데이터의 rmse오차 
      test_predict = sess.run(hypothesis, feed_dict={X: testX})
      test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
      test_error_summary.append(test_error)
        
      # print current loss
      print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch+1, 
                                                                               train_error, 
                                                                               test_error, 
                                                                               train_error-test_error))
        

  # print result
  for i in range(len(test_predict)):
    predict = reverse_min_max_scaling(val_prices, test_predict)
    answer = reverse_min_max_scaling(val_prices, testY)  
    print(answer[i], predict[i])


  # save result
  answer_file = open('val_answer.txt', 'w')
  for element in answer:
    an = str(element)
    answer_file.write(an+'\n')

  answer_file.close()
  
  predict_file = open('val_predict.txt', 'w')
  for element in predict:
    pr = str(element)
    predict_file.write(pr+'\n')
  predict_file.close()

if __name__ == "__main__":
  tf.app.run()
