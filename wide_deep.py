from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='./census_model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=2, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='./census_data/adult.data',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='./census_data/adult.test',
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

INPUT_COLUMNS = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', [' Female', ' Male']),

    tf.feature_column.categorical_column_with_vocabulary_list(
        'race',
        [' Amer-Indian-Eskimo', ' Asian-Pac-Islander',
         ' Black', ' Other', ' White']
    ),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'education',
        [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
         ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
         ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th',
         ' 1st-4th', ' Preschool', ' 12th']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',
        [' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
         ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        [' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
         ' Other-relative']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',
        [' Self-emp-not-inc', ' Private', ' State-gov',
         ' Federal-gov', ' Local-gov', ' ?', ' Self-emp-inc',
         ' Without-pay', ' Never-worked']
    ),

    # For columns with a large number of values, or unknown values
    # We can use a hash function to convert to categories.
    tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket(
        'native_country', hash_bucket_size=100, dtype=tf.string),

    # Continuous base columns.
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education_num'),
    tf.feature_column.numeric_column('capital_gain'),
    tf.feature_column.numeric_column('capital_loss'),
    tf.feature_column.numeric_column('hours_per_week'),
]

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  (gender, race, education, marital_status, relationship,
   workclass, occupation, native_country, age,
   education_num, capital_gain, capital_loss, hours_per_week) = INPUT_COLUMNS
  # Build an estimator.

  # Reused Transformations.
  # Continuous columns can be converted to categorical via bucketization
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [
      # Interactions between different categorical features can also
      # be added as new virtual features.
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=int(1e4)),
      tf.feature_column.crossed_column(
          [age_buckets, race, 'occupation'], hash_bucket_size=int(1e6)),
      tf.feature_column.crossed_column(
          ['native_country', 'occupation'], hash_bucket_size=int(1e4)),
      gender,
      native_country,
      education,
      occupation,
      workclass,
      marital_status,
      relationship,
      age_buckets,
  ]

  embedding_size = 8
  deep_columns = [
      # Use indicator columns for low dimensional vocabularies
      tf.feature_column.indicator_column(workclass),
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(gender),
      tf.feature_column.indicator_column(relationship),
      tf.feature_column.indicator_column(race),

      # Use embedding columns for high dimensional vocabularies
      tf.feature_column.embedding_column(
          native_country, dimension=embedding_size),
      tf.feature_column.embedding_column(occupation, dimension=embedding_size),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]
  return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    return features, tf.equal(labels, '>50K')

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

def eval_input_fn(features, labels, batch_size):
  """An input function for evaluation or prediction"""
  features=dict(features)
  if labels is None:
    # No labels, use only features.
    inputs = features
  else:
    inputs = (features, labels)

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices(inputs)

  # Batch the examples
  assert batch_size is not None, "batch_size must not be None"
  dataset = dataset.batch(batch_size)

  # Return the dataset.
  return dataset

def predict_instance(model):
  predict_x = { 
  'age':[30],
  'gender': ['Male'],
  'workclass':["State-gov"],
  'fnlwgt':[141297],
  'education':["Bachelors"],
  'education_num':[13],
  'marital_status':["Married-civ-spouse"],
  'occupation':["Prof-specialty"],
  'relationship':["Husband"],
  'capital_gain':[0],
  'capital_loss':[0],
  'hours_per_week':[40],
  }
  for idx in range(len(_CSV_COLUMNS)):
    k = _CSV_COLUMNS[idx]
    if k not in predict_x:
      v = _CSV_COLUMN_DEFAULTS[idx]
      predict_x[k] = v
  
  predictions = model.predict(input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=100),)
  print(str(predictions))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print("c:" + str(class_id))
    print("p:" + str(probability))


def save_model(model, model_dir="./model"):
  feature_spec = tf.feature_column.make_parse_example_spec(INPUT_COLUMNS)
  serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
  model.export_savedmodel(model_dir , serving_input_receiver_fn, as_text=True)

def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    model.train(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

    results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))

  save_model(model)
  predict_instance(model)


"""
show input and output name with cmd:
saved_model_cli show --dir /SAVED_MODEL_DIR  --all
"""

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

