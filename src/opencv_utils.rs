use crate::Result;
use core::slice::from_raw_parts;
use ndarray::prelude::*;
use opencv::core::{ValidVecType, Vec3};
use opencv::prelude::*;

pub fn convert_mat_to_array2<T>(mat: &Mat) -> Result<ArrayView2<T>>
where
    T: ValidVecType + DataType,
{
    Ok(ArrayView::from_shape(
        (mat.rows() as usize, mat.cols() as usize),
        mat.data_typed::<T>()?,
    )?)
}

pub fn convert_mat_to_array3<'a, T>(mat: &'a Mat) -> Result<ArrayView3<'a, T>>
where
    T: ValidVecType + DataType,
    Vec3<T>: DataType,
{
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels()? as usize;
    // SAFETY: It's safe here because `Mat` is contiguous in memory.
    let slice: &'a [T] = unsafe {
        from_raw_parts(
            mat.data_typed::<Vec3<T>>()?.as_ptr().cast::<T>(),
            rows * cols * channels,
        )
    };

    Ok(ArrayView::from_shape((rows, cols, channels), slice)?)
}

pub fn convert_image_mat_to_array3(image: &Mat) -> Result<ArrayView3<u8>> {
    let mut array = convert_mat_to_array3(image)?;
    // Convert BGR to RGB
    array.invert_axis(Axis(2));

    Ok(array)
}

pub fn convert_array2_to_mat<T: DataType>(array: ArrayView2<T>) -> Result<Mat> {
    Ok(
        // `unwrap()` is safe here
        Mat::from_slice(array.as_standard_layout().as_slice().unwrap())?
            .reshape(1, array.shape()[0] as i32)?,
    )
}

pub fn convert_array3_to_mat<T: DataType>(array: ArrayView3<T>) -> Result<Mat> {
    let shape = array.shape();
    Ok(
        // `unwrap()` is safe here
        Mat::from_slice(array.as_standard_layout().as_slice().unwrap())?
            .reshape(shape[2] as i32, shape[0] as i32)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert() -> Result<()> {
        use opencv::videoio::{VideoCapture, CAP_ANY};

        let mut cap = VideoCapture::from_file("examples/assets/kira.gif", CAP_ANY)?;
        let mut frame = Mat::default();

        loop {
            if !cap.read(&mut frame)? {
                break;
            }

            let array1 = convert_image_mat_to_array3(&frame)?;
            let mat = convert_array3_to_mat(array1)?;
            let array2: ArrayView3<u8> = convert_mat_to_array3(&mat)?;
            assert_eq!(array1, array2);
        }

        let array1: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
        let mat = convert_array2_to_mat(array1.view())?;
        let array2: ArrayView2<i32> = convert_mat_to_array2(&mat)?;
        assert_eq!(array1, array2);

        Ok(())
    }
}
