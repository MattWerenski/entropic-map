# entropic-map
Code accompanying the paper *Estimation of entropy-regularized optimal transport maps between
non-compactly supported measures* on [on Arxiv](TODO)

## Set up

In order to run this library you will need a python environment with

```numpy scipy matplotlib pot```

## Strucuture

`entropic_maps.py` includes most of the interesting code and implements the estimator discussed in the paper.

`testing.py` is a file which contains basic functions for running experiments using random samples and aggregating the results.

`tests/[test_name]` are files for generating datasets which are used to make the plots contained in the paper.

`plotting/plot_[test_name]` takes in the output files of `test/[test_name]` and actually makes the plots. 


From inside either `tests/` or `plotting/` you can run the scripts. The scripts do not need to be modified except for specifying where the datasets are to be stored to / loaded from. 
