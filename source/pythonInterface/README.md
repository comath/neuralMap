# Cython Wrappers

These are the cython wrappers to expose the C utilities to python. There are 3 main interfaces we can use, `ipTrace`, `nnLayersUtilsWrap` and 'mapperWrap'. `mapperWrap` inherits `ipTrace` which inherits `nnLayersUtilsWrap`. The `keyWrap` is to expose the conversion tools as by default the C files and the wraps return the bit packed signatures. 