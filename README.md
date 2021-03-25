# RobustStereoMatching
To download checkpoint files for our method, go to https://drive.google.com/drive/folders/1OWNqiLUY4n-RfhOFJzxLYbQv8SUCwvGx?usp=sharing and download all files to checkpoint/MCTNet/


To test our trained model:<br />
python test.py --left_image=examples/left.png --right_image=examples/right.png --backbone=1

Not setting --backbone=1 will use the one without backbone.

To train and evaluate our model, please download from:<br />
1. SceneFlow - https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html<br />
2. KITTI2015 - http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

Our default directory setting is:<br />
data/<br />
  FlyingThings3D/<br />
    frames_finalpass/<br />
    disparity/<br />
  KITTI2015/<br />
    training/<br />
    testing/<br />
projects/<br />
  RobustStereoMatching/<br />
    train.py<br />
    ...
    
To train our model:<br />
 ./train.sh
 
To evaluate model (dataset: 1 - SceneFlow, 3 - KITTI2015):<br />
  python evaluation.py --whichModel=2 --dataset=1 --backbone=1
 
To test other models such as GANet (https://github.com/feihuzhang/GANet) or PSMNet (https://github.com/JiaRenChang/PSMNet), please go to their repo and put it in the folder models. GANet will need to be compiled.
 
To test stereo-constrained attacks:<br />
  python pgd_attack.py --whichModel=2 --dataset=1 --backbone=1 --e=0.03 --total_iter=20
 
To test synthetic patch attacks:<br />
  python synthetic_patch_attack.py<br />
Use --test_patch_shift to test generated patch at different disparity levels.
