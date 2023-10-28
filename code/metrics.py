"""
File: metrics.py
Author: jzx-bupt
"""
import numpy as np


def previous_accuracy(proba, y_test, max_page_num):
    true_count_list = [0] * max_page_num
    for i, sample in enumerate(proba):
        for j, page in enumerate(sample):
            if np.argmax(page) == np.argmax(y_test[i][j]):
                true_count_list[j] += 1
    accuracy = [round(true_count / len(proba), 3) for true_count in true_count_list]
    print("Accuracy on each page:", accuracy)
    return accuracy


def overall_basic_accuracy(proba, y_test, class_num):
    total_count = 0
    true_count = 0
    for i, sample in enumerate(proba):
        true_set = set([np.argmax(page) for page in y_test[i]])
        true_set.discard(class_num - 1)  # remove non-object class
        pred_set = set([np.argmax(page) for page in sample])
        pred_set.discard(class_num - 1)  # remove non-object class
        total_count += max(len(true_set), len(pred_set))
        true_count += len(true_set & pred_set)
    accuracy = round(float(true_count / total_count), 3)

    print("Total accuracy(loose):", accuracy)

    return accuracy


def overall_advanced_accuracy(proba, y_test, class_num):
    total_count = 0
    true_count = 0
    for i, sample in enumerate(proba):
        true_list = [np.argmax(page) for page in y_test[i]]
        pred_list = [np.argmax(page) for page in sample]

        for true_lb, pred_lb in zip(true_list, pred_list):

            if pred_lb != true_lb:
                total_count += 1
            if true_lb != class_num - 1:
                if pred_lb == true_lb:
                    true_count += 1
                    total_count += 1
    accuracy = round(float(true_count / total_count), 3)
    print("Total accuracy(strict):", accuracy)

    return accuracy


def previous_precision(proba, y_test, class_num, max_page_num):
    tp_list = [[0 for _ in range(class_num)] for _ in range(max_page_num)]
    fp_list = [[0 for _ in range(class_num)] for _ in range(max_page_num)]
    for i, sample in enumerate(proba):
        for j in range(max_page_num):
            true_label = int(np.argmax(y_test[i][j]))
            pred_label = int(np.argmax(sample[j]))
            if pred_label == true_label:
                tp_list[j][true_label] += 1
            else:
                fp_list[j][pred_label] += 1
    tp_plus_fp_list = [np.array(tp_list[i]) + np.array(fp_list[i]) for i in range(max_page_num)]
    for i, page in enumerate(tp_plus_fp_list):
        # The denominator array may contain 0 elements, resulting in result overflow
        tp_plus_fp_list[i] = np.array([e if e != 0 else 1 for e in page])
    precision_list = [np.array(tp) / tp_plus_fp for tp, tp_plus_fp in zip(tp_list, tp_plus_fp_list)]
    avg_pre_list = [round(float(np.mean(pre)), 3) for pre in precision_list]
    print("Average Precision on each page:", avg_pre_list)
    return avg_pre_list


def overall_basic_precision(proba, y_test, class_num):
    tp_list = [0 for _ in range(class_num - 1)]
    fp_list = [0 for _ in range(class_num - 1)]
    for i, sample in enumerate(proba):
        true_set = set([np.argmax(page) for page in y_test[i]])
        true_set.discard(class_num - 1)  # remove non-object class
        pred_set = set([np.argmax(page) for page in sample])
        pred_set.discard(class_num - 1)  # remove non-object class
        for label in pred_set:
            if label in true_set:
                tp_list[label] += 1
            else:
                fp_list[label] += 1
    tp_plus_fp_list = [tp + fp for tp, fp in zip(tp_list, fp_list)]
    tp_plus_fp_list = [e if e != 0 else 1 for e in tp_plus_fp_list]  # The denominator array may contain 0 elements, resulting in result overflow
    precision = np.array(tp_list) / np.array(tp_plus_fp_list)
    precision = round(float(np.mean(precision)), 3)
    print("Total precision(loose):", precision)

    return precision


def overall_advanced_precision(proba, y_test, class_num):
    tp_list = [0 for _ in range(class_num - 1)]
    fp_list = [0 for _ in range(class_num - 1)]
    for i, sample in enumerate(proba):
        true_list = [np.argmax(page) for page in y_test[i]]
        pred_list = [np.argmax(page) for page in sample]
        for true_lb, pred_lb in zip(true_list, pred_list):
            if true_lb != class_num - 1 or pred_lb != class_num - 1:
                if true_lb == pred_lb:
                    tp_list[true_lb] += 1
                else:
                    if pred_lb != class_num - 1:
                        fp_list[pred_lb] += 1
    tp_plus_fp_list = [tp + fp for tp, fp in zip(tp_list, fp_list)]
    tp_plus_fp_list = [e if e != 0 else 1 for e in tp_plus_fp_list]  # The denominator array may contain 0 elements, resulting in result overflow
    precision = np.array(tp_list) / np.array(tp_plus_fp_list)
    precision = round(float(np.mean(precision)), 3)
    print("Total precision(strict):", precision)

    return precision


def previous_recall(proba, y_test, class_num, max_page_num):
    tp_list = [[0 for _ in range(class_num)] for _ in range(max_page_num)]
    fn_list = [[0 for _ in range(class_num)] for _ in range(max_page_num)]
    for i, sample in enumerate(proba):
        for j in range(max_page_num):
            true_label = int(np.argmax(y_test[i][j]))
            pred_label = int(np.argmax(sample[j]))
            if pred_label == true_label:
                tp_list[j][true_label] += 1
            else:
                fn_list[j][true_label] += 1
    tp_plus_fn_list = [np.array(tp_list[i]) + np.array(fn_list[i]) for i in range(max_page_num)]
    for i, page in enumerate(tp_plus_fn_list):
        # The denominator array may contain 0 elements, resulting in result overflow
        tp_plus_fn_list[i] = np.array([e if e != 0 else 1 for e in page])
    recall_list = [np.array(tp) / tp_plus_fn for tp, tp_plus_fn in zip(tp_list, tp_plus_fn_list)]
    avg_rec_list = [round(float(np.mean(rec)), 3) for rec in recall_list]
    print("Average Recall on each page:", avg_rec_list)
    return avg_rec_list


def overall_basic_recall(proba, y_test, class_num):
    tp_list = [0 for _ in range(class_num - 1)]
    fn_list = [0 for _ in range(class_num - 1)]
    for i, sample in enumerate(proba):
        true_set = set([np.argmax(page) for page in y_test[i]])
        true_set.discard(class_num - 1)  # remove non-object class
        pred_set = set([np.argmax(page) for page in sample])
        pred_set.discard(class_num - 1)  # remove non-object class
        for label in true_set:
            if label in pred_set:
                tp_list[label] += 1
            else:
                fn_list[label] += 1
    tp_plus_fn_list = [tp + fn for tp, fn in zip(tp_list, fn_list)]
    tp_plus_fn_list = [e if e != 0 else 1 for e in tp_plus_fn_list]  # The denominator array may contain 0 elements, resulting in result overflow
    recall = np.array(tp_list) / np.array(tp_plus_fn_list)
    recall = round(float(np.mean(recall)), 3)
    print("Total recall(loose):", recall)
    return recall


def overall_advanced_recall(proba, y_test, class_num):
    tp_list = [0 for _ in range(class_num - 1)]
    fn_list = [0 for _ in range(class_num - 1)]
    for i, sample in enumerate(proba):
        true_list = [np.argmax(page) for page in y_test[i]]
        pred_list = [np.argmax(page) for page in sample]
        for true_lb, pred_lb in zip(true_list, pred_list):
            if true_lb != class_num - 1 or pred_lb != class_num - 1:
                if true_lb == pred_lb:
                    tp_list[true_lb] += 1
                else:
                    if true_lb != class_num - 1:
                        fn_list[true_lb] += 1
    tp_plus_fn_list = [tp + fn for tp, fn in zip(tp_list, fn_list)]
    tp_plus_fn_list = [e if e != 0 else 1 for e in tp_plus_fn_list]  # The denominator array may contain 0 elements, resulting in result overflow
    recall = np.array(tp_list) / np.array(tp_plus_fn_list)
    recall = round(float(np.mean(recall)), 3)
    print("Total recall(strict):", recall)
    return recall