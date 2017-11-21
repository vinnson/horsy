from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import tempfile
import itertools

import pandas as pd
from six.moves import urllib
import tensorflow as tf

CSV_COLUMNS = [
    "id","season","race_no","date","place","rc","track","course",
    "distance","grass","class","dr","position","act_wt","wt","time","gear",
    "lbw","trainer_code","jockey_code","win_odd","rating","horse_code","win_odd_order",
    "rating_order","act_wt_order","wt_order","place_3"
]

rc = tf.contrib.layers.sparse_column_with_keys(column_name="rc", keys=["ST","HV"])
track = tf.contrib.layers.sparse_column_with_keys(column_name="track", keys=["Turf","AWT"])
course = tf.contrib.layers.sparse_column_with_keys(column_name="course", keys=["-", "A", "A+3", "B", "B+2", "C", "C+3"])
grass = tf.contrib.layers.sparse_column_with_keys(column_name="grass",
    keys=["FT", "G", "GD", "GF", "GY", "S", "SL", "WF", "WS", "Y", "YS"])

# To show an example of hashing:
gear = tf.feature_column.categorical_column_with_hash_bucket("gear", hash_bucket_size=1000)
trainer_code = tf.feature_column.categorical_column_with_hash_bucket("trainer_code", hash_bucket_size=1000)
jockey_code = tf.feature_column.categorical_column_with_hash_bucket("jockey_code", hash_bucket_size=1000)
horse_code = tf.feature_column.categorical_column_with_hash_bucket("horse_code", hash_bucket_size=1000)
hclass = tf.feature_column.categorical_column_with_hash_bucket("class", hash_bucket_size=100)

# Continuous base columns.
race_no = tf.feature_column.numeric_column("race_no")
distance = tf.feature_column.numeric_column("distance")
dr = tf.feature_column.numeric_column("dr")
rating = tf.feature_column.numeric_column("rating")
#act_wt = tf.feature_column.numeric_column("act_wt")
#wt = tf.feature_column.numeric_column("wt")
#win_odd = tf.feature_column.numeric_column("win_odd")

# Transformations.
#age_buckets = tf.feature_column.bucketized_column(
#    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Wide columns and deep columns.
base_columns = [
    race_no, track, course, distance, grass, hclass, dr, rating,
    gear, trainer_code, jockey_code, horse_code
]

crossed_columns = [
    tf.feature_column.crossed_column(["race_no", "distance", "class"], hash_bucket_size=1000),
    #tf.feature_column.crossed_column(["trainer_code", "jockey_code", "horse_code"], hash_bucket_size=2000),
    #tf.feature_column.crossed_column(["distance", "horse_code", "dr", "class"], hash_bucket_size=2000),
    #tf.feature_column.crossed_column(
    #    [distance, dr], hash_bucket_size=1000),
    #tf.feature_column.crossed_column(
    #    ["native_country", "occupation"], hash_bucket_size=1000)
]

deep_columns = [
    race_no,
    distance,
    #dr,
    #rating,
    #tf.feature_column.embedding_column(track, dimension=8),
    #tf.feature_column.embedding_column(course, dimension=8),
    #tf.feature_column.embedding_column(grass, dimension=8),
    # To show an example of embedding
    tf.feature_column.embedding_column(hclass, dimension=8),
    tf.feature_column.embedding_column(trainer_code, dimension=8),
    tf.feature_column.embedding_column(jockey_code, dimension=8),
    #tf.feature_column.embedding_column(horse_code, dimension=8),
]

def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data["class"] = str(df_data["class"])
  df_data = df_data.dropna(how="any", axis=0)
  #labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
  labels = df_data["place_3"].apply(lambda x: int(x) == 1).astype(int)
  #labels = df_data["place"].apply(lambda x: "1" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)

def build_estimator(model_dir):
  """Build an estimator."""
  m = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=crossed_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
  return m

def train_and_eval():
  model_dir = tempfile.mkdtemp()
  m = build_estimator(model_dir)
  # set num_epochs to None to get infinite stream of data.
  m.train(
      input_fn=input_fn("/opt/tensor/race_result_clean.csv", num_epochs=None, shuffle=True),
      steps=2000)
  # set steps to None to run evaluation until all data consumed.
  results = m.evaluate(
      input_fn=input_fn("/opt/tensor/race_result.csv", num_epochs=1, shuffle=False),
      steps=None)
  print("model directory = %s" % model_dir)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))
  print ("----------------- START PREDICT -----------------")
  y = m.predict(
    input_fn=input_fn("/opt/tensor/race_result.csv", num_epochs=1, shuffle=True))
  predictions = list(p for p in itertools.islice(y, 105))
  for p in predictions:
      print (p)
      sys.exit()
  # Manual cleanup
  shutil.rmtree(model_dir)

FLAGS = None

def main(_):
  train_and_eval()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=2000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
