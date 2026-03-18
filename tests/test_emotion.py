"""
tests/test_emotion.py — 情緒偵測模組測試（不需真實語音，用正弦波模擬）

執行：pytest tests/test_emotion.py -v
"""
import numpy as np
import pytest
from src.emotion.classifier import EmotionLabel, EmotionSVM, generate_synthetic_training_data
from src.emotion.feature_extractor import AudioFeatures, EmotionFeatureExtractor


def make_sine(freq=440.0, duration=0.5, sr=22050) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestAudioFeatures:
    def test_feature_vector_shape(self):
        f = AudioFeatures(np.zeros(40), np.zeros(40), np.zeros(40),
                          np.zeros(5), np.zeros(5), 1.0, 22050)
        assert f.feature_vector.shape == (130,)
        assert f.n_features == 130

    def test_to_dict_keys(self):
        f = AudioFeatures(np.ones(40), np.zeros(40), np.zeros(40),
                          np.array([200.,20.,150.,280.,130.]),
                          np.array([0.05,0.02,0.01,0.1,0.09]), 2.5, 22050)
        d = f.to_dict()
        for k in ["mfcc_mean","pitch_mean","energy_mean","duration_s"]:
            assert k in d


class TestEmotionFeatureExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        return EmotionFeatureExtractor(sample_rate=22050)

    def test_extract_shape(self, extractor):
        y = make_sine(220., 1.0)
        f = extractor.extract(y, sr=22050)
        assert f.mfcc.shape == (40,)
        assert f.pitch_stats.shape == (5,)
        assert f.duration_s == pytest.approx(1.0, abs=0.05)

    def test_realtime_stats_keys(self, extractor):
        y = make_sine(440., 0.2)
        s = extractor.extract_realtime_stats(y, sr=22050)
        assert "energy_rms" in s and "tension_index" in s
        assert 0.0 <= s["tension_index"] <= 1.0

    def test_nonzero_mfcc(self, extractor):
        y = make_sine(300., 0.5)
        f = extractor.extract(y, sr=22050)
        assert not np.allclose(f.mfcc, 0)

    def test_silent_low_energy(self, extractor):
        silent = np.zeros(22050, dtype=np.float32)
        f = extractor.extract(silent, sr=22050)
        assert f.energy_stats[0] < 1e-4


class TestEmotionSVM:
    @pytest.fixture(scope="class")
    def trained(self):
        X, y = generate_synthetic_training_data(60)
        clf = EmotionSVM(C=10.)
        clf.train(X, y, test_size=0.2)
        return clf

    def test_model_exists(self, trained):
        assert trained._model is not None
        assert len(trained._classes) == 4

    def test_predict_valid_label(self, trained):
        X, _ = generate_synthetic_training_data(5)
        p = trained.predict(X[0])
        assert p.label in list(EmotionLabel)
        assert 0. <= p.confidence <= 1.

    def test_proba_sum_one(self, trained):
        X, _ = generate_synthetic_training_data(5)
        p = trained.predict(X[1])
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_save_load(self, trained, tmp_path):
        path = tmp_path / "svm.pkl"
        trained.save(str(path))
        loaded = EmotionSVM.load(str(path))
        X, _ = generate_synthetic_training_data(3)
        assert trained.predict(X[0]).label == loaded.predict(X[0]).label

    def test_calm_not_comfort(self):
        assert EmotionLabel.CALM.needs_comfort is False
        assert EmotionLabel.ANXIOUS.needs_comfort is True


class TestEmotionDetector:
    @pytest.mark.asyncio
    async def test_no_model_fallback(self):
        from src.emotion.detector import EmotionDetector
        det = EmotionDetector(model_path=None)
        y = make_sine(440., 0.5)
        pred = await det.detect(y, sr=22050)
        assert pred.label in list(EmotionLabel)
        assert isinstance(pred.confidence, float)
