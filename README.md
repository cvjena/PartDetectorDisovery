# Part Detector Discovery

This is the code used in our paper "Part Detector Discovery in Deep Convolutional Neural Networks" by Marcel Simon, Erik Rodner and Joachim Denzler published at ACCV 2014. 
If you would like to refer to this work, please cite the corresponding paper

    @inproceedings{Simon14:PDD,
      author = {Marcel Simon and Erik Rodner and Joachim Denzler},
      booktitle = {Asian Conference on Computer Vision (ACCV)},
      title = {Part Detector Discovery in Deep Convolutional Neural Networks},
      year = {2014},
    }

The following steps will guide you through the usage of the code.

## 1. Python Environment
Setup a python environment, preferably a virtual environment using e. g. virtual_env. The requirements file might install more than you need. 

```
virtualenv pyhton-env && pip install -r requirements.txt
```

## 2. DeCAF Installation
Build and install decaf into this environment

```
source python-env/bin/activate
cd decaf-tools/decaf/
python setup.py build
python setup.py install
```

## 3. Pre-Trained ImageNet Model

Get the decaf ImageNet model:

```
cd decaf-tools/models/
bash get_model.sh
```

You now might need to adjust the path to the decaf model in decaf-tools/extract_grad_map.py, line 75!

## 4. Gradient Map Calculation

Now you can calculate the gradient maps using the following command. For a single image, use decaf-tools/extract_grad_map.py :

```
usage: extract_grad_map.py [-h] [--layers LAYERS [LAYERS ...]] [--limit LIMIT]
                           [--channel_limit CHANNEL_LIMIT]
                           [--images pattern [pattern ...]] [--outdir OUTDIR]

Calculate the gradient maps for an image.

optional arguments:
  -h, --help            show this help message and exit
  --layers LAYERS [LAYERS ...]
  --limit LIMIT         When calculating the gradient of the class scores,
                        calculate the gradient for the output elements with the
                        [limit] highest probabilities.
  --channel_limit CHANNEL_LIMIT
                        Sets the number of channels per layer you want to
                        calculate the gradient of.
  --images pattern [pattern ...]
			Absolute image path to the image. You can use wildcards.
  --outdir OUTDIR
```

For a list of absolute image paths call this script this way:

```
python extract_grad_map.py --images $(cat /path/to/imagelist.txt) --limit 1 --channel_limit 256 --layers probs pool5 --outdir /path/to/output/
```

The gradient maps are stored as Matlab .mat file and as png. In addition to these, the script also generates A html file to view the gradient maps and the input image. The gradient map is placed in the directory outdir/images'_parent_dir/image_filename/*. Be aware that approx. 45 MiB of storage is required per input image. For the whole CUB200-2011 dataset this means a total storage size of approx 800 GiB!

## 5. Part Localization

Apply the part localization using GMM fitting or maximum finding. Have a look in the part_localization folder for that. Open calcCUBPartLocs.m and adjust the paths. Now simply run calcCUBPartLocs(). This will create a file which has the same format as the part_locs.txt file of the CUB200-2011 dataset. You can use it for part-based classification. 

## 6. Classification


We also provide the classification framework to use these part localizations and feature extraction with DeCAF. Go to the folder classification and open partEstimationDeepLearing.m. Have a look at line 40 and adjust the path such that it points to the correct file. Open settings.m and adjust the paths. Next, open settings.m and adjust the paths to liblinear and the virtual python environment. Now you can execute for example:

```
init
recRate = experimentParts('cub200_2011',200, struct('descriptor','plain','preprocessing_useMask','none','preprocessing_cropToBoundingbox',0), struct('partSelection',[1 2 3 9 14],'bothSymmetricParts',0,'descriptor','plain','trainPartLocation','est','preprocessing_relativePartSize',1.0/8,'preprocessing_cropToBoundingbox',0))
```

This will evaluate the classification performance on the standard train-test-split using the estimated part locations. Experiment parts has four parameters. The first one tell the function which dataset to use. You want to keep `'cub200_2011'` here. 

The second one is the number of classes to use, `3`, `14` and `200` is supported here. Next is the setup for the global feature extraction. The only important setting is `preprocessing_cropToBoundingbox`. A value of `0` will tell the function not to use the ground truth bounding box during testing. You should leave the other two options as shown here. 

The last one is the setup for the part features. You can select here, which parts you want to use and if you want to extract features from both symmetric parts, if both are visible. Since the part detector discovery associates some parts with the same channel, the location prediction will be the same for these. In this case, only select the parts which have unique channels here. In the example, the part 1, 2, 3, 9 and 14 are associated with different channels. 

`'trainPartLocation'` tells the function, if grount-truth (`'gt'`) or estimated (`'est'`) part locations should be used for training. Since the discovered part detectors do not necessarily relate to semantic parts, `'est'` usually is the better option here. 

`'preprocessing_relativePartSize'` adjusts the size of patches, that are extracted at the estimated part locations. Please have a look at the paper for more information. 

For the remaining options, you should keep everything as it is. 

## Acknowledgements
The classification framework is an extension of the excellent fine-grained recognition framework by Christoph Göring, Erik Rodner, Alexander Freytag and Joachim Denzler. You can find their project at [https://github.com/cvjena/finegrained-cvpr2014](https://github.com/cvjena/finegrained-cvpr2014).

Our work is based on DeCAF, a framework for convolutional neural networks. You can find the repository of the corresponding project at [https://github.com/UCB-ICSI-Vision-Group/decaf-release/ ](https://github.com/UCB-ICSI-Vision-Group/decaf-release/).

## License 
Part Detector Discovery Framework by [Marcel Simon](http://www.inf-cv.uni-jena.de/simon.html), [Erik Rodner](http://www.inf-cv.uni-jena.de/rodner.html) and [Joachim Denzler](http://www.inf-cv.uni-jena.de/denzler.html) is licensed under the non-commercial license [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/). For usage beyond the scope of this license, please contact [Marcel Simon](http://www.inf-cv.uni-jena.de/simon.html).