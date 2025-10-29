import pytest

@pytest.mark.xfail(reason="Need to debug initialization")
def test_pipeline_initialization():
    """Test pipeline component initialization"""
    from scripts.data_acquisition import DataAcquisition
    from scripts.preprocessing import Preprocessor
    from scripts.anomaly_detection import AnomalyDetector

    data_acquisition = DataAcquisition(aws_credentials={"aws_access_key_id": "test"})
    preprocessor = Preprocessor()
    anomaly_detector = AnomalyDetector(thresholds={"file_size_max": 10_000_000})

    assert isinstance(data_acquisition, DataAcquisition)
    assert isinstance(preprocessor, Preprocessor)
    assert isinstance(anomaly_detector, AnomalyDetector)