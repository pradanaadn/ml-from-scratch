from metric.implementation.ce import cross_entropy

def test_cross_entropy_basic():
    y_true = [1, 0, 0]
    y_pred = [0.7, 0.2, 0.1]
    result = round(cross_entropy(y_true, y_pred), 6)
    expected = round(0.1549019599857432, 6)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_cross_entropy_all_zeros():
    y_true = [0, 0, 0]
    y_pred = [0.3, 0.3, 0.4]
    result = round(cross_entropy(y_true, y_pred), 6)
    expected = round(0.0, 6)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_cross_entropy_all_ones():
    y_true = [1, 1, 1]
    y_pred = [0.3, 0.3, 0.4]
    result = round(cross_entropy(y_true, y_pred), 6)
    expected = round(1.443697, 6)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_cross_entropy_mixed():
    y_true = [1, 0, 1]
    y_pred = [0.2, 0.5, 0.3]
    result = round(cross_entropy(y_true, y_pred), 6)
    expected = round(1.221849, 6)
    assert result == expected, f"Expected {expected}, but got {result}"