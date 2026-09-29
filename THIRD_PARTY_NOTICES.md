# Third-party notices

## SalsaNext (model and KNN post-processing)

`src/as25lm/segmentation/salsanext.py` and `src/as25lm/segmentation/knn.py` are taken from
[tiagoCortinhal/SalsaNext](https://github.com/tiagoCortinhal/SalsaNext)
(`train/tasks/semantic/modules/SalsaNext.py` and `train/tasks/semantic/postproc/KNN.py`).
Change: removed unused imports for Python 3.12. The pretrained weights are from the same project.

```
MIT License

Copyright (c) 2019 Tiago Cortinhal (Halmstad University, Sweden), George Tzelepis (Volvo Technology AB,
Volvo Group Trucks Technology, Sweden) and Eren Erdal Aksoy (Halmstad University and Volvo Technology AB, Sweden)

Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab,
University of Bonn.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Data

KITTI odometry scans and SemanticKITTI labels are not included in this repository. They are
published by their authors under their own terms (non-commercial research use); download them
from the official sources.
