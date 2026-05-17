SMC_MIREX Readme

Hi,

In the archive there are 4 folders:

SMC_MIREX_Annotations 
SMC_MIREX_Annotations_05_08_2014 
SMC_MIREX_Audio  
SMC_MIREX_Tags 

They contain the following:

SMC_MIREX_Annotations
---------------------

Ground truth beat annotations in seconds
The filenames are a bit confusing, their structure is not so important, but to explain:

SMC_001_2_1_1_a.txt means 

file: SMC_001

other metrical interpretation: 2_1_1 (beats tapped in a 2:1 ratio, starting on beat 1 could also be acceptable) - NOTE this is not tested nor should it be used!

a: gives the name of the annotator (again this information is not useful)

SMC_MIREX_Annotations_05_08_2014
--------------------------------

The content of this directory is the same as above, except for the annotations for excerpts 056, 137, 153, 203 and 257 which have been updated to remove the final beat annotations which were out of range. Thanks to Andy Lambert for pointing this out. 


SMC_MIREX_Audio
---------------

The audio files (mono .wav at 44.1khz)
There are 217 in total, but they are numbered running up to 289.
Note, files 271 - 289 are "easy" compared to the rest which are hard.

SMC_MIREX_Tags
--------------

These are text files with tags that correspond to why the annotation was difficult, along with a code, e.g. f1 where 'f' is the name of the annotator and '1' is a confidence level (again this information is not very relevant for you).
These tags are probably not so useful, but might be good for post-hoc analysis of results.

Acknowledgment
--------------
If you use the dataset in your work, please cite the following paper:

Holzapfel, A.; Davies, M.E.P.; Zapata, J.R.; Oliveira, J.L.; Gouyon, F.; , "Selective Sampling for Beat Tracking Evaluation," Audio, Speech, and Language Processing, IEEE Transactions on , vol.20, no.9, pp.2539-2548, Nov. 2012
doi: 10.1109/TASL.2012.2205244
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6220849&isnumber=6268383


Any questions, send me a mail: mdavies@inesctec.pt


