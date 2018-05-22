import tensorflow as tf
import re
from collections import Counter
import string

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def get_record_parser(config, is_test=False):
    """
    Get the tfrecords sample parser.
    :param config: Contains the configurations to be used.
    :param is_test: Indicate if the data_type is test.
    :return: The parser method.
    """
    def parse(example):
        """
        Extract features from a single tfrecords sample.
        :param example: Contain the example.
        :return: context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
        """
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(
            features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(
            features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(
            features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(
            features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    """
    Get a random batch for the next training phase.
    :param record_file:
    :param parser: The tfrecords parser of the saved format.
    :param config: Contains the configurations to be used.
    :return: The batch to train on.
    """
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    """
    Get the saved dataset to be fed into the model.
    :param record_file: The tfrecords file contained the process inputs.
    :param parser: The tfrecords parser of the saved format.
    :param config: Contains the configurations to be used.
    :return: The dataset object.
    """
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    """
    Convert the predictions from the model into the format to be used by
      the evaluation function.
    :param eval_file: Contains the correct answers.
    :param qa_id: Question ids
    :param pp1: Start positions.
    :param pp2: End positions.
    :return: answer_dict - The answers later evaluated.
            remapped_dict - The remapped answer to be saved into file on test mode.
    """
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        # The beginning of the start span.
        start_idx = spans[p1][0]
        # The ending of the last span.
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    """
    Calculate model performance over given predictions.
    :param eval_file: Contains the correct answers.
    :param answer_dict: The predictions to evaluate.
    :return: dictionary of the evaluated scores (exact_match, f1).
    """
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    """
    Normalize an answer.
    :param s: The answer sentence to normalize.
    :return: The normalized sentence.
    """
    def remove_articles(text):
        """
        Clear any article and replace it with empty space.
        :param text: The text to filter.
        :return: The filtered text.
        """
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        """
        Normalized any spaces into a single space.
        :param text: The text to normalize.
        :return: The normalized text.
        """
        return ' '.join(text.split())

    def remove_punc(text):
        """
        Eliminate any punctuation from the text.
        :param text: The text to filter.
        :return: The filtered text.
        """
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        """
        Lower all letters.
        :param text: The text to normalized.
        :return: The normalized text.
        """
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Calcualte the F1 score between prediction and ground_truth.
    :param prediction: The prediction to evaluate.
    :param ground_truth: The ground truth to consider.
    :return: The score value.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Check if the prediction is exactly matched the ground_truth
    :param prediction:
    :param ground_truth:
    :return: Boolean indicator.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Calculate the max prediction score of prediction against the given ground_truths.
    :param metric_fn: A metric function to evaluate the prediction
                        against the ground_truths.
    :param prediction: The prediction to evaluate.
    :param ground_truths: The ground_truths to consider.
    :return: The max score value.
    """
    max_score = 0
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        if score > max_score:
            max_score = score
    return max_score
