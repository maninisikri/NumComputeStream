def test_pipeline_partial_fit_runs():
    from numcompute_stream.pipeline import Pipeline
    from numcompute_stream.preprocessing import StandardScaler
    from numcompute_stream.tree import DecisionTreeClassifier
    import numpy as np
    X = np.array([[1,2],[3,4],[5,6]], dtype=float)
    y = np.array([0,0,1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier()),
    ])
    pipe.partial_fit(X, y)
    assert pipe._fitted

def test_pipeline_partial_fit_no_labels():
    from numcompute_stream.pipeline import Pipeline
    from numcompute_stream.preprocessing import StandardScaler
    import numpy as np
    X = np.array([[1,2],[3,4],[5,6]], dtype=float)
    pipe = Pipeline([("scaler", StandardScaler())])
    pipe.partial_fit(X)
    assert pipe._fitted