
Work log:

* Implemented basic demo from template
* Implemented boundary detection using CHD measure (can't cope with fade in/out)
* Added groundtruth labels for boundary detection
    -- see data/*boundaries.gt.csv
* Added groundtruth labels for scene classification
    -- see data/*scenes.gt.csv
    -- data/*scenes.in* is a more general groundtruth file that could be used to generate the desired output format
* Manually identified the CHD thresholds
    -- see results below

* First algorithm variant loaded all images at once and computed everything in a batch
* Started implementing boundary detection using edge ratios
* Refactored code to become a sequential algorithm
* Refactored code by adding Frame and Keyframe classes
* Refactored code by adding ShotBoundaryDetectorCHD, ShotBoundaryDetectorEdgeRatio and SceneMatcher classes


TODOs:
    - TODO: plot histograms for debugging purposes
    -- ideally in debug mode windows pop up that display histograms.
    -- code for histogram plots exists, but wasn't sure if should stay with opencv or with matplotlib
    -- Ideally the visualisation of the plots looks similar to the output in the paper + including groundtruth to quickly compare the results and look up thresholds

    - TODO: move color histogram computation code from ShotBoundaryDetectorEdgeRatio into SceneMatcherSceneMatcher
    
    - TODO: ShotBoundaryDetectorEdgeRatio: Detect fades by looking at previous frames
    -- the paper describes how to detect fades, but this is still a TODO

    - TODO: SceneMatcher: treat fades differently by looking at start and end frame and potentially merging/replacing previously detected hard keyframes
    -- once we can identify fades, the scene matcher will need to update previous keyframes if a fade is detected.
    -- Possibly a hard cut will be replaced by 

    - TODO: evaluation framework for csv files created with --evaluate switch
    -- Ideally the evaluation doesn't need to be done manually, but the threshold is chosen that maximises accuracy automatically, for example when considering the classification tasks as binary problems (correct scene/wrong scene) or using multi-class confusion matrices




Command line examples:

$ python demo.py data/frames
INFO:    * Read input sequence from directory: data/frames
INFO:    * Found 985 files
INFO:    * Analysing image sequence
INFO:    Shot boundary     0 starts at keyframe 000001.jpg
INFO:    Shot boundary     1 starts at keyframe 000003.jpg
INFO:    Shot boundary     2 starts at keyframe 000062.jpg
INFO:    Shot boundary     3 starts at keyframe 000103.jpg
INFO:    Shot boundary     4 starts at keyframe 000163.jpg
INFO:    Shot boundary     3 starts at keyframe 000221.jpg
INFO:    Shot boundary     4 starts at keyframe 000277.jpg
INFO:    Shot boundary     3 starts at keyframe 000323.jpg
INFO:    Shot boundary     4 starts at keyframe 000371.jpg
INFO:    Shot boundary     2 starts at keyframe 000398.jpg
INFO:    Shot boundary     3 starts at keyframe 000449.jpg
INFO:    Shot boundary     4 starts at keyframe 000478.jpg
INFO:    Shot boundary     2 starts at keyframe 000569.jpg
INFO:    Shot boundary     5 starts at keyframe 000652.jpg
INFO:    * Found 14/5 unique shots
INFO:    * Write data/frames.boundaries.csv
INFO:    * Write data/frames.scenes.csv
INFO:    * Algorithm took 6.93 seconds for   985 frames (142.08 FPS)



Should create data/frames.scenes.csv:

"# keyframe, scene id"
000001.jpg,A
000003.jpg,B
000062.jpg,C
000103.jpg,D
000163.jpg,E
000221.jpg,D
000277.jpg,E
000323.jpg,D
000371.jpg,E
000398.jpg,C
000449.jpg,D
000478.jpg,E
000569.jpg,C
000652.jpg,F

In comparison to data/frames.scenes.gt.csv:
# keyframe, scene id
000001.jpg,A
000003.jpg,B
000062.jpg,C
000103.jpg,D
000163.jpg,E
000221.jpg,D
000277.jpg,E
000323.jpg,D
000371.jpg,E
000398.jpg,F
000449.jpg,G
000478.jpg,E
000569.jpg,F
000652.jpg,H
000722.jpg,I
000747.jpg,I/J
000756.jpg,J






$ python demo.py data/frames2
INFO:    * Read input sequence from directory: data/frames2
INFO:    * Found 2771 files
INFO:    * Analysing image sequence
INFO:    Shot boundary     0 starts at keyframe 000001.jpg
INFO:    Shot boundary     1 starts at keyframe 000100.jpg
INFO:    Shot boundary     2 starts at keyframe 000324.jpg
INFO:    Shot boundary     1 starts at keyframe 000433.jpg
INFO:    Shot boundary     0 starts at keyframe 000633.jpg
INFO:    Shot boundary     3 starts at keyframe 000816.jpg
INFO:    Shot boundary     0 starts at keyframe 000943.jpg
INFO:    Shot boundary     0 starts at keyframe 001346.jpg
INFO:    Shot boundary     0 starts at keyframe 001773.jpg
INFO:    Shot boundary     4 starts at keyframe 002648.jpg
INFO:    Shot boundary     5 starts at keyframe 002649.jpg
INFO:    Shot boundary     6 starts at keyframe 002652.jpg
INFO:    Shot boundary     7 starts at keyframe 002653.jpg
INFO:    Shot boundary     8 starts at keyframe 002654.jpg
INFO:    Shot boundary     9 starts at keyframe 002655.jpg
INFO:    * Found 15/9 unique shots
INFO:    * Write data/frames2.boundaries.csv
INFO:    * Write data/frames2.scenes.csv
INFO:    * Algorithm took 34.91 seconds for  2771 frames (79.38 FPS)




Should create data/frames2.scenes.csv:

"# keyframe, scene id"
000001.jpg,A
000100.jpg,B
000324.jpg,C
000433.jpg,B
000633.jpg,A
000816.jpg,D
000943.jpg,A
001346.jpg,A
001773.jpg,A
002648.jpg,E
002649.jpg,F
002652.jpg,G
002653.jpg,H
002654.jpg,I
002655.jpg,J


In comparison to the groundtruth data/frames2.scenes.gt.csv:

# keyframe, scene id
000001.jpg,A
000100.jpg,B
000324.jpg,C
000433.jpg,B
000633.jpg,A
000816.jpg,D
000838.jpg,D/B
000853.jpg,B
000943.jpg,A
001236.jpg,B
001346.jpg,A
001575.jpg,B
001773.jpg,E
001817.jpg,E/B
001832.jpg,B
001893.jpg,B/F
001912.jpg,F
002193.jpg,F/D
002245.jpg,D
002648.jpg,G




$ python demo.py data/frames2 --evaluate

will additionally create the files
- evaluation.chd.plot.csv
- evaluation.chisqr_dist.plot.csv
which can be used for evaluation purpose
