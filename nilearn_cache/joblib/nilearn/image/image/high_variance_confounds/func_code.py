# first line: 62
def high_variance_confounds(
    imgs, n_confounds=5, percentile=2.0, detrend=True, mask_img=None
):
    """Return confounds extracted from input signals with highest variance.

    Parameters
    ----------
    imgs : Niimg-like object
        4D image.
        See :ref:`extracting_data`.

    mask_img : Niimg-like object
        If not provided, all voxels are used.
        If provided, confounds are extracted from voxels inside the mask.
        See :ref:`extracting_data`.

    n_confounds : :obj:`int`, default=5
        Number of confounds to return.

    percentile : :obj:`float`, default=2
        Highest-variance signals percentile to keep before computing the
        singular value decomposition, 0. <= `percentile` <= 100.
        `mask_img.sum() * percentile / 100` must be greater than `n_confounds`.

    detrend : :obj:`bool`, default=True
        If True, detrend signals before processing.

    Returns
    -------
    :class:`numpy.ndarray`
        Highest variance confounds. Shape: *(number_of_scans, n_confounds)*.

    Notes
    -----
    This method is related to what has been published in the literature
    as 'CompCor' (Behzadi NeuroImage 2007).

    The implemented algorithm does the following:

    - Computes the sum of squares for each signal (no mean removal).
    - Keeps a given percentile of signals with highest variance (percentile).
    - Computes an SVD of the extracted signals.
    - Returns a given number (n_confounds) of signals from the SVD with
      highest singular values.

    See Also
    --------
    nilearn.signal.high_variance_confounds

    """
    from .. import masking

    if mask_img is not None:
        sigs = masking.apply_mask(imgs, mask_img)
    else:
        # Load the data only if it doesn't need to be masked
        imgs = check_niimg_4d(imgs)
        sigs = as_ndarray(get_data(imgs))
        # Not using apply_mask here saves memory in most cases.
        del imgs  # help reduce memory consumption
        sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T

    return signal.high_variance_confounds(
        sigs, n_confounds=n_confounds, percentile=percentile, detrend=detrend
    )
