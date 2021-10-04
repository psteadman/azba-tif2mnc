#!/usr/bin/env python

from pyminc.volumes.factory import *
import numpy as n
import scipy.misc
import scipy.interpolate
import scipy.ndimage
from optparse import OptionParser, OptionGroup

import numpy as n
import scipy.interpolate
import scipy.ndimage

# taken from http://www.scipy.org/Cookbook/Rebinning
def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [n.float64, n.float32]:
        a = n.cast[float](a)

    m1 = n.cast[int](minusone)
    ofs = n.cast[int](centre) * 0.5
    old = n.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = n.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = n.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = n.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = n.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = n.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = n.mgrid[nslices]

        newcoords_dims = range(n.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (n.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported."
        return None


if __name__ == "__main__":
    usage = "%prog [options] in_1.tif in_2.tif ..- in_n.tif out.mnc"
    description = """
%prog takes a series of two-dimensional images (tif, png, jpeg
 ... whatever can be read by python's PIL library) and converts them
 into a 3D MINC volume. Additional options control for resampling of
 slices - in the case of histology data, for example, the 2D slices
 might be at much higher resolution than the final desired 3D
 volume. In that case the slices are preblurred and then downsampled
 using nearest neighbour resampling before being inserted into the 3D
 volume.
"""
    
    parser = OptionParser(usage=usage, description=description)

    size_group = OptionGroup(parser, "Determining input and output size")
    size_group.add_option("--input-resolution", dest="input_resolution",
                          help="Input resolution in mm (i.e. the pixel size "
                          "assuming that it's isotropic) [default: %default]",
                          type="float", default=0.00137)
    size_group.add_option("--output-resolution", dest="output_resolution",
                          help="The desired output resolution in mm (i.e. "
                          "what the data will be resampled to) "
                          "[default: %default]",
                          type="float", default=0.075)
    size_group.add_option("--slice-gap", dest="slice_gap",
                          help="The slice gap in mm (i.e. the distance "
                          "between adjacent slices)[default: %default]",
                          type="float", default=0.075)
    # add option for explicitly giving output matrix size
    parser.add_option_group(size_group)

    preprocessing_group = OptionGroup(parser, "Preprocessing of each slice")
    preprocessing_group.add_option("--gaussian", action="store_const",
                                   help="Apply 2D gaussian (FWHM set based "
                                   "on input and output sizes)",
                                   const="gaussian", dest="preprocess",
                                   default=None)
    preprocessing_group.add_option("--uniform", action="store_const",
                                   help="Apply 2D uniform filter (size based "
                                   "on input and output sizes)",
                                   const="uniform", dest="preprocess")
    preprocessing_group.add_option("--uniform-sum", action="store_const",
                                   help="Apply 2D uniform filter and multiply "
                                   "it by filter volume to obtain a count. Use "
                                   "this option if, for example, the slices "
                                   "contain classified neurons.",
                                   const="uniform_sum", dest="preprocess")
    parser.add_option_group(preprocessing_group)
    
    dim_group = OptionGroup(parser, "Volume dimension options")
    dim_group.add_option("--xyz", action="store_const",
                         help="XYZ dimension order",
                         const=("xspace", "yspace", "zspace"),
                         dest="dimorder",
                         default=("yspace", "zspace", "xspace"))
    dim_group.add_option("--xzy", action="store_const",
                         help="XZY dimension order",
                         const=("xspace", "zspace", "yspace"),
                         dest="dimorder")
    dim_group.add_option("--yxz", action="store_const",
                         help="YXZ dimension order",
                         const=("yspace", "xspace", "zspace"),
                         dest="dimorder")
    dim_group.add_option("--yzx", action="store_const",
                         help="YZX dimension order [default]",
                         const=("yspace", "zspace", "xspace"),
                         dest="dimorder")
    dim_group.add_option("--zxy", action="store_const",
                         help="ZXY dimension order",
                         const=("zspace", "xspace", "yspace"),
                         dest="dimorder")
    dim_group.add_option("--zyx", action="store_const",
                         help="ZYX dimension order",
                         const=("zspace", "yspace", "xspace"),
                         dest="dimorder")

    parser.add_option_group(dim_group)

    (options, args) = parser.parse_args()
    
    # construct volume
    # need to know the number of slices
    output_filename = args.pop()
    n_slices = len(args)
    # need to know the size of the output slices - read in a single slice
    test_slice = scipy.misc.imread(args[0])
    slice_shape = n.array(test_slice.shape)
    size_fraction = options.input_resolution / options.output_resolution
    output_size = n.ceil(slice_shape * size_fraction).astype('int')
    filter_size = n.ceil(slice_shape[0] / output_size[0])

    vol = volumeFromDescription(output_filename, 
                                options.dimorder,
                                sizes=(n_slices,output_size[0],output_size[1]),
                                starts=(0,0,0),
                                steps=(options.slice_gap, 
                                       options.output_resolution,
                                       options.output_resolution), 
                                volumeType='ushort')
    for i in range(n_slices):
        print "In slice", i+1, "out of", n_slices
        imslice = scipy.misc.imread(args[i])

        # normalize slice to lie between 0 and 1
        original_type_max = n.iinfo(imslice.dtype).max
        imslice = imslice.astype('float')
        imslice = imslice / original_type_max

        # smooth the data depending on the chosen option
        if options.preprocess=="gaussian":
            imslice = scipy.ndimage.gaussian_filter(imslice, sigma=filter_size)
        if options.preprocess=="uniform" or options.preprocess=="uniform_sum":
            imslice = scipy.ndimage.uniform_filter(imslice, size=filter_size)
        if options.preprocess=="uniform_sum":
            imslice = imslice * filter_size * filter_size

        # downsample the slice
        o_imslice = congrid(imslice, output_size, 'neighbour')
        # add the downsampled slice to the volume
        vol.data[i,:,:] = o_imslice

    # finish: write the volume to file
    vol.writeFile()
    vol.closeVolume()
