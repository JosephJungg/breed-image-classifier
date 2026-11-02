import numpy as np

from src.analysis import cat_dog_mixups, confused_pairs, per_class_accuracy, report

CLASSES = ["Birman", "Ragdoll", "Beagle"]


def test_per_class_accuracy_sorted_worst_first():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2])
    rows = per_class_accuracy(y_true, y_pred, CLASSES)
    assert rows[0] == ("Birman", 0.5, 4)
    assert {r[0] for r in rows[1:]} == {"Ragdoll", "Beagle"}


def test_per_class_accuracy_skips_absent_classes():
    rows = per_class_accuracy(np.array([0, 0]), np.array([0, 1]), CLASSES)
    assert [r[0] for r in rows] == ["Birman"]


def test_confused_pairs_keep_direction():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([1, 1, 0, 2])
    pairs = confused_pairs(y_true, y_pred, CLASSES)
    assert pairs[("Birman", "Ragdoll")] == 2
    assert pairs[("Ragdoll", "Birman")] == 1
    assert sum(pairs.values()) == 3


def test_cat_dog_mixups_ignore_same_species_errors():
    y_true = np.array([0, 0, 2])
    y_pred = np.array([1, 2, 0])     # cat->cat, cat->dog, dog->cat
    assert cat_dog_mixups(y_true, y_pred, CLASSES) == 2


def test_report_mentions_totals():
    text = report(np.array([0, 1, 2, 2]), np.array([1, 1, 2, 2]), CLASSES)
    assert "1 mistakes of 4" in text
    assert "1x  Birman" in text
    assert "cat/dog mix-ups: 0" in text
