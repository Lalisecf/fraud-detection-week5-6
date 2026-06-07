import pandas as pd

def test_dataframe_creation():
    df = pd.DataFrame({"A":[1,2,3]})
    assert len(df) == 3
