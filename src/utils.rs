use crate::{OpenVtuberError, Rectangles, Result};
use core::cmp::Ord;
use core::ops::Deref;
use image::{ImageBuffer, Pixel};
use ndarray::prelude::*;
use ndarray::{DataMut, DataOwned, RawData, ScalarOperand};
use std::ops::{Add, Div, Mul, Sub};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, Interpreter, InterpreterBuilder};

pub fn clip<A, S, D>(mut array: ArrayBase<S, D>, min: A, max: A) -> ArrayBase<S, D>
where
    A: Copy + Ord,
    S: DataMut + RawData<Elem = A>,
    D: Dimension,
{
    array.mapv_inplace(|a| a.max(min).min(max));
    array
}

pub fn clip_f32<S, D>(mut array: ArrayBase<S, D>, min: f32, max: f32) -> ArrayBase<S, D>
where
    S: DataMut + RawData<Elem = f32>,
    D: Dimension,
{
    array.mapv_inplace(|a| a.max(min).min(max));
    array
}

pub(crate) fn build_interpreter(filepath: &str) -> Result<Interpreter<BuiltinOpResolver>> {
    let model = FlatBufferModel::build_from_file(filepath)?;
    let resolver = BuiltinOpResolver::default();
    let builder = InterpreterBuilder::new(model, resolver)?;
    let mut interpreter = builder.build()?;
    interpreter.allocate_tensors()?;

    Ok(interpreter)
}

/// Non-Maximum Suppression
///
/// # Panics
/// Panics if the scores contain `NaN`.
pub fn non_max_suppression(boxes: Rectangles, scores: Vec<f32>, iou_threshold: f32) -> Rectangles {
    let areas = boxes
        .iter()
        .map(|b| (b.x2 - b.x1) * (b.y2 - b.y1))
        .collect::<Vec<_>>();
    let mut zips = boxes.into_iter().zip(areas).zip(scores).collect::<Vec<_>>();
    zips.sort_unstable_by(|((_, _), s1), ((_, _), s2)| {
        // `NaN` can cause panic here.
        s1.partial_cmp(s2).expect("failed to compare `NaN`")
    });
    let mut picked = Vec::new();

    while let Some(p) = zips.pop() {
        picked.push(p);
        let mut reserved = Vec::new();
        for z in zips {
            let width = (p.0 .0.x2.min(z.0 .0.x2) - p.0 .0.x1.max(z.0 .0.x1)).max(0.);
            let height = (p.0 .0.y2.min(z.0 .0.y2) - p.0 .0.y1.max(z.0 .0.y1)).max(0.);
            let intersection = width * height;
            let ratio = intersection / (p.0 .1 + z.0 .1 - intersection);
            if ratio < iou_threshold {
                reserved.push(z);
            }
        }

        reserved.sort_unstable_by(|((_, _), s1), ((_, _), s2)| {
            // `unwrap()` is safe here.
            s1.partial_cmp(s2).unwrap()
        });
        zips = reserved;
    }

    picked
        .into_iter()
        .map(|p| p.0 .0)
        .collect::<Vec<_>>()
        .into()
}

/// Normalize
pub fn normalize_min_max<A, S, D>(array: ArrayBase<S, D>, min: A, max: A) -> Result<ArrayBase<S, D>>
where
    A: Copy
        + PartialOrd
        + ScalarOperand
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>,
    S: DataOwned<Elem = A> + DataMut,
    D: Dimension,
{
    let first = array
        .first()
        .ok_or(OpenVtuberError::Error("the input array is empty"))?;
    let min_value = *array
        .iter()
        .try_fold(first, |m, a| {
            m.partial_cmp(a).map(|o| match o {
                std::cmp::Ordering::Less => m,
                std::cmp::Ordering::Equal => m,
                std::cmp::Ordering::Greater => a,
            })
        })
        .ok_or(OpenVtuberError::Error(
            "failed to compare two numbers in the array",
        ))?;
    let max_value = *array
        .iter()
        .try_fold(first, |m, a| {
            m.partial_cmp(a).map(|o| match o {
                std::cmp::Ordering::Less => a,
                std::cmp::Ordering::Equal => m,
                std::cmp::Ordering::Greater => m,
            })
        })
        .ok_or(OpenVtuberError::Error(
            "failed to compare two numbers in the array",
        ))?;

    Ok((array - min_value) / (max_value - min_value) * (max - min) + min)
}

pub fn convert_array3_to_image<A, P>(array: ArrayView3<A>) -> Result<ImageBuffer<P, &[A]>>
where
    A: 'static,
    P: 'static + Pixel<Subpixel = A>,
{
    let slice = array.to_slice().ok_or(OpenVtuberError::Error(
        "failed to convert the array to a slice",
    ))?;
    let shape = array.shape();

    ImageBuffer::from_raw(shape[1] as u32, shape[0] as u32, slice)
        .ok_or(OpenVtuberError::Error("failed to contruct `ImageBuffer`"))
}

pub fn convert_image_to_array3<A, P, Container>(
    image: &ImageBuffer<P, Container>,
) -> Result<ArrayView3<A>>
where
    A: 'static,
    P: 'static + Pixel<Subpixel = A>,
    Container: Deref<Target = [A]>,
{
    let (width, height) = image.dimensions();
    let slice = image.as_raw().deref();

    Ok(ArrayView::from_shape(
        (height as usize, width as usize, P::CHANNEL_COUNT as usize),
        slice,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip() {
        let array = array![10, -10, 2];
        assert_eq!(clip(array, -5, 5), array![5, -5, 2]);

        let array = array![10., -10., 2.];
        assert_eq!(clip_f32(array, -5., 5.), array![5., -5., 2.]);
    }

    #[test]
    fn test_normalize() -> Result<()> {
        let array = normalize_min_max(array![2., 6., -4.], -1., 1.)?;
        assert_eq!(array, array![0.19999999999999996, 1., -1.]);

        Ok(())
    }

    #[test]
    fn test_convert_image() -> Result<()> {
        use crate::opencv_utils::convert_image_mat_to_array3;
        use image::Rgb;
        use opencv::prelude::*;
        use opencv::videoio::{VideoCapture, CAP_ANY};

        let mut cap = VideoCapture::from_file("examples/assets/kira.gif", CAP_ANY)?;
        let mut frame = Mat::default();
        cap.read(&mut frame)?;

        let array = convert_image_mat_to_array3(&frame)?;
        let array = array.as_standard_layout();
        let image: ImageBuffer<Rgb<_>, _> = convert_array3_to_image(array.view()).unwrap();
        let array_view = convert_image_to_array3(&image)?;
        assert_eq!(array_view, array);

        Ok(())
    }
}
