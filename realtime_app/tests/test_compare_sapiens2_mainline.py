import numpy as np
import pytest
import cv2

from tools.compare_sapiens2_mainline import assert_frozen_prompts, sapiens_points, saved_pair


def test_frozen_prompts_compare_actual_float32_model_inputs():
    box = [0.,0.,1079.,1907.3245760074135]
    source = {'detections':[{'bbox_xyxy':box,'score':.8}]}
    saved = {'boxes_from_yolo26x':np.asarray([box],dtype=np.float32).tolist(),
             'bbox_scores_from_yolo26x':np.asarray([.8],dtype=np.float32).tolist()}
    assert assert_frozen_prompts(source,saved) == 1
    saved['boxes_from_yolo26x'][0][3] += .01
    with pytest.raises(ValueError,match='prompts differ'):
        assert_frozen_prompts(source,saved)


def test_sapiens_mapping_preserves_anatomical_wrist_and_lower_limb_order():
    xy = np.column_stack((np.arange(308),1000+np.arange(308)))
    scores = np.linspace(0.,1.,308)
    result = sapiens_points({'keypoints308':xy,'keypoint_scores':scores})
    assert result.shape == (17,3)
    assert result[9,0] == 62 and result[10,0] == 41
    np.testing.assert_array_equal(result[11:,0],np.arange(9,15))
    np.testing.assert_array_equal(result[:,2],scores[[0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14]])


def test_mapping_preserves_out_of_sensor_coordinates_and_rejects_nonfinite():
    item = {'keypoints308':np.ones((308,2)),'keypoint_scores':np.ones(308)}
    item['keypoints308'][62] = [-12.,2200.]
    np.testing.assert_array_equal(sapiens_points(item)[9,:2],[-12.,2200.])
    item['keypoints308'][62,0] = np.nan
    with pytest.raises(ValueError,match='invalid Sapiens'):
        sapiens_points(item)


def test_frozen_box_score_change_is_not_silently_accepted():
    entry = {'detections':[{'bbox_xyxy':[0.,0.,20.,30.],'score':.8}]}
    prediction = {'boxes_from_yolo26x':[[0.,0.,20.,30.]],'bbox_scores_from_yolo26x':[.7]}
    with pytest.raises(ValueError,match='prompts differ'):
        assert_frozen_prompts(entry,prediction)


def test_saved_pair_preserves_metadata_and_inverse_rotates_without_resizing(tmp_path, monkeypatch):
    image = np.arange(6*4*3,dtype=np.uint8).reshape(6,4,3)
    def read(path):
        assert path.endswith('pair_0017.png')
        assert 'left_ccw90' in path or 'right_cw90' in path
        return image.copy()
    monkeypatch.setattr(cv2,'imread',read)
    row = {'pair_id':17,'file_name':'pair_0017.png','timestamp_skew_ms':12.,'timestamp_type':'frozen_source',
           'left':{'source_frame_id':31,'source_timestamp_sec':10.},
           'right':{'source_frame_id':35,'source_timestamp_sec':10.012}}
    pair = saved_pair(row,{'input_dir':str(tmp_path)})
    assert pair.pair_id == 17 and pair.left.frame_id == 31 and pair.right.frame_id == 35
    assert pair.left.timestamp_sec == 10. and pair.right.timestamp_sec == 10.012
    assert pair.timestamp_skew_sec == .012 and pair.timestamp_type == 'frozen_source'
    np.testing.assert_array_equal(pair.left.image,cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))
    np.testing.assert_array_equal(pair.right.image,cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE))
