import numpy as np
from core.algorithms import scores, segmentation, preprocess, clean_mask, overlay
from core.unet_tp4 import predict_busi, predict_busi_outputs, preprocess_busi_like_tp
from core.tp3 import segment_tp3

def test_scores_exact():
    mask=np.array([[0,255],[255,0]],dtype=np.uint8)
    assert scores(mask,mask)=={"Dice":1.0,"IoU":1.0}
    assert scores(np.zeros_like(mask),mask)=={"Dice":0.0,"IoU":0.0}

def test_methods_shapes():
    im=np.zeros((56,80),dtype=np.uint8)
    im[12:39,20:60]=180
    for method in ["Otsu","Seuil manuel","Watershed"]:
        result=segmentation(im,method)
        assert result.shape==im.shape,method
        assert set(np.unique(result)).issubset({0,255}),method
    tp3_im=im.copy()
    tp3_im[20:30,30:45]=90  # TP3 K-means is fixed at three classes.
    rgb=np.stack([tp3_im]*3,-1)
    for method in ["K-means","Mean Shift","Chan–Vese"]:
        _,result,_=segment_tp3(rgb,method)
        assert result.ndim==2,method
        assert set(np.unique(result)).issubset({0,255}),method

def test_preprocessing_and_overlay():
    im=np.zeros((80,100),dtype=np.uint8)
    for m in ("Aucun","Médian","Gaussien","Bilatéral","CLAHE","Égalisation histogramme"):
        x=preprocess(im,m)
        assert x.shape==im.shape
    assert overlay(np.stack([im]*3,-1),im).shape==(80,100,3)
    assert clean_mask(im).shape==im.shape

def test_tp4_normalization_equivalence():
    im=np.arange(64*72,dtype=np.uint8).reshape(64,72)
    import cv2
    resized=cv2.resize(im,(256,256),interpolation=cv2.INTER_LINEAR)
    expected=resized.astype(np.float32)[None,:,:]
    expected/=np.maximum(np.linalg.norm(expected,axis=1,keepdims=True),1e-12)
    actual=preprocess_busi_like_tp(im)
    assert actual.shape==(1,256,256,1)
    np.testing.assert_allclose(actual[:,:,:,0],expected,atol=1e-7)


def test_tp4_prediction_keeps_probability_and_uses_strict_threshold():
    class FakeModel:
        def predict(self, batch, verbose=0):
            assert batch.shape == (1, 256, 256, 1)
            prediction = np.zeros((1, 256, 256, 1), dtype=np.float32)
            prediction[:, :, :128, 0] = 0.5
            prediction[:, :, 128:, 0] = 0.5001
            return prediction

    image = np.arange(40 * 60, dtype=np.uint8).reshape(40, 60)
    probability, mask = predict_busi_outputs(FakeModel(), image)
    assert probability.shape == image.shape
    assert mask.shape == image.shape
    assert set(np.unique(mask)) == {0, 255}
    assert np.all(mask[:, :30] == 0)  # Exactly 0.5 must remain background.
    assert np.all(mask[:, 30:] == 255)
    np.testing.assert_array_equal(predict_busi(FakeModel(), image), mask)
