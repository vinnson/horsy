# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/tmp/model", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "tmp/tmpIw3Z5G",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "tmp/tmpqbMhzx",
    "Path to the test data.")

COLUMNS = ["id", "season", "raceno", "date", "place",
  "rc","track","racecourse","racedistance","racegrass","raceclass",
  "racedr","position","actwt","wt","time","gear","lbw",
  "trainer_code","thejockeycode","win_odd","rating","thehorsecode",
  "win_odd_order","rating_order","act_wt_order","wt_order","place_3"]

LABEL_COLUMN = "place"
CATEGORICAL_COLUMNS = []
CONTINUOUS_COLUMNS = []

def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  season = tf.contrib.layers.sparse_column_with_hash_bucket("season", 100)
  raceno = tf.contrib.layers.sparse_column_with_hash_bucket("raceno", 10)
  rc = tf.contrib.layers.sparse_column_with_hash_bucket("rc", 2)
  track = tf.contrib.layers.sparse_column_with_hash_bucket("track", 2)
  racecourse = tf.contrib.layers.sparse_column_with_hash_bucket("racecourse", 20)
  racedistance = tf.contrib.layers.sparse_column_with_hash_bucket("racedistance", 20)
  racegrass = tf.contrib.layers.sparse_column_with_hash_bucket("racegrass", 20)
  raceclass = tf.contrib.layers.sparse_column_with_hash_bucket("raceclass", 100)
  racedr = tf.contrib.layers.sparse_column_with_hash_bucket("racedr", 20)
  actwt = tf.contrib.layers.sparse_column_with_hash_bucket("actwt", 1000)
  wt = tf.contrib.layers.sparse_column_with_hash_bucket("wt", 1000)
  gear = tf.contrib.layers.sparse_column_with_hash_bucket("gear", 50)
  trainer_code = tf.contrib.layers.sparse_column_with_hash_bucket("trainer_code", 1000)
  thejockeycode = tf.contrib.layers.sparse_column_with_hash_bucket("thejockeycode", 1000)
  win_odd = tf.contrib.layers.sparse_column_with_hash_bucket("win_odd", 1000)
  rating = tf.contrib.layers.sparse_column_with_hash_bucket("rating", 100)
  thehorsecode = tf.contrib.layers.sparse_column_with_hash_bucket("thehorsecode", 1000)

  # Wide columns and deep columns.
  wide_columns = [season, raceno, rc, track, racecourse, racedistance, racegrass, raceclass, racedr, actwt, wt, gear,
				  trainer_code, thejockeycode, win_odd, rating, thehorsecode,
				  tf.contrib.layers.crossed_column([raceno, rc, track, racecourse, racedistance, racegrass, raceclass, racedr], hash_bucket_size=int(1e4)),
				  tf.contrib.layers.crossed_column([trainer_code, thejockeycode, thehorsecode], hash_bucket_size=int(1e4))]

  deep_columns = [season, raceno, rc, track, racecourse, racedistance, racegrass, raceclass, racedr, actwt, wt, gear,
				  trainer_code, thejockeycode, win_odd, rating, thehorsecode]

  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[10, 20,40,20,10])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[10, 20,40,20,10])
  return m

def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval():
  """Train and evaluate the model."""
 # train_file_name = 'adult.data'
 # test_file_name = 'adult.test'
  train_file_name = 'poker-hand-testing.data'
  test_file_name = 'poker-hand-training-true.data'
  #test_file_name = maybe_download()
  df_train = pd.read_csv(
      tf.gfile.Open("/opt/tensor/race_result_clean.csv"),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1)
  df_test = pd.read_csv(
      tf.gfile.Open("/opt/tensor/race_result_clean.csv"),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1)

  #df_train[LABEL_COLUMN] = (df_train["CLASS_Poker_Hand"].apply(lambda x: x>5)).astype(int)
  #df_test[LABEL_COLUMN] = (df_test["CLASS_Poker_Hand"].apply(lambda x: x>5)).astype(int)

  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)
  m = build_estimator(model_dir)
  print(m)
  m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))



def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
