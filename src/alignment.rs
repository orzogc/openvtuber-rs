use crate::utils::{build_interpreter, convert_image_to_array3};
use crate::{Input, Points, Rectangle, Result};
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};
use ndarray::prelude::*;
use ndarray::{concatenate, Data, RawDataClone};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::Interpreter;

pub struct FaceAlignment<'a> {
    interpreter: Interpreter<'a, BuiltinOpResolver>,
    marker_nums: i32,
    input_size: [i32; 2],
    trans_distance: f32,
    input_index: i32,
    output_index: i32,
}

impl<'a> FaceAlignment<'a> {
    pub fn get_landmarks<S>(
        &mut self,
        image: &Input<S>,
        face_box: &Rectangle<f32>,
    ) -> Result<Points>
    where
        S: Data<Elem = u8> + RawDataClone,
    {
        let (inp, i_m) = self.pre_processing(image, face_box)?;
        let out = self.inference(inp)?;

        Ok(self.post_processing(out, i_m)?)
    }

    fn pre_processing<S>(
        &self,
        image: &Input<S>,
        face_box: &Rectangle<f32>,
    ) -> Result<(Array4<f32>, Array2<f32>)>
    where
        S: Data<Elem = u8> + RawDataClone,
    {
        let maximum_edge = (face_box.x2 - face_box.x1).max(face_box.y2 - face_box.y1) * 3.;
        let scale = self.trans_distance * 4. / maximum_edge;
        let inverse_scale = 1. / scale;
        let cx = self.trans_distance - scale * (face_box.x1 + face_box.x2) / 2.;
        let cy = self.trans_distance - scale * (face_box.y1 + face_box.y2) / 2.;

        let inverse_matrix = array![
            [inverse_scale, 0., -cx * inverse_scale],
            [0., inverse_scale, -cy * inverse_scale]
        ];

        let scaled_width = (image.width() as f32 * scale).round() as usize;
        let scaled_height = (image.height() as f32 * scale).round() as usize;
        let scaled_image: ImageBuffer<Rgb<_>, _> =
            image.resize(scaled_width, scaled_height, FilterType::Nearest)?;
        let scaled_image = convert_image_to_array3(&scaled_image)?;

        let mut translation: Array3<f32> =
            ArrayBase::zeros((self.input_size[0] as usize, self.input_size[1] as usize, 3));

        for w in 0..scaled_image.shape()[1] {
            let w_t = (w as f32 + cx).round();
            if w_t < 0. || w_t >= self.input_size[0] as f32 {
                continue;
            }
            let w_t = w_t as usize;
            for h in 0..scaled_image.shape()[0] {
                let h_t = (h as f32 + cy).round();
                if h_t < 0. || h_t >= self.input_size[1] as f32 {
                    continue;
                }
                let h_t = h_t as usize;
                translation
                    .slice_mut(s![h_t, w_t, ..])
                    .iter_mut()
                    .zip(scaled_image.slice(s![h, w, ..]))
                    .for_each(|(t, s)| *t = *s as f32);
            }
        }

        Ok((translation.insert_axis(Axis(0)), inverse_matrix))
    }

    fn inference(&mut self, input: Array4<f32>) -> Result<Vec<f32>> {
        self.set_input(input)?;
        self.interpreter.invoke()?;
        Ok(self.get_output()?.to_vec())
    }

    fn post_processing(&self, out: Vec<f32>, i_m: Array2<f32>) -> Result<Points> {
        let col: Array2<f32> = ArrayBase::ones((self.marker_nums as usize, 1));
        let mut out: Array2<f32> = ArrayBase::from_shape_vec((self.marker_nums as usize, 2), out)?;
        out += 1.;
        out *= self.trans_distance;
        out = concatenate![Axis(1), out, col];

        Ok(out
            .dot(&i_m.t())
            .outer_iter()
            .map(|o| o.as_standard_layout().as_slice().unwrap().into()) // `unwrap()` is safe here.
            .collect::<Vec<_>>()
            .into())
    }

    fn set_input(&mut self, input: Array4<f32>) -> Result<()> {
        let input_tensor: &mut [f32] = self.interpreter.tensor_data_mut(self.input_index)?;
        input_tensor
            .iter_mut()
            .zip(&input)
            .for_each(|(it, i)| *it = *i);

        Ok(())
    }

    fn get_output(&self) -> Result<&[f32]> {
        Ok(self.interpreter.tensor_data(self.output_index)?)
    }
}

#[derive(Clone, Debug)]
pub struct FaceAlignmentBuilder {
    filepath: String,
    marker_nums: i32,
    input_size: [i32; 2],
}

impl FaceAlignmentBuilder {
    #[inline]
    pub fn new(filepath: impl Into<String>) -> Self {
        FaceAlignmentBuilder {
            filepath: filepath.into(),
            marker_nums: 106,
            input_size: [192, 192],
        }
    }

    #[inline]
    pub fn with_marker_nums(&mut self, marker_nums: i32) -> &mut Self {
        self.marker_nums = marker_nums;
        self
    }

    #[inline]
    pub fn with_input_size(&mut self, width: i32, height: i32) -> &mut Self {
        self.input_size = [width, height];
        self
    }

    pub fn build(&self) -> Result<FaceAlignment> {
        let trans_distance = self.input_size[1] as f32 / 2.;

        let interpreter = build_interpreter(self.filepath.as_str())?;
        let input_index = interpreter.inputs()[0];
        let output_index = interpreter.outputs()[0];

        Ok(FaceAlignment {
            interpreter,
            marker_nums: self.marker_nums,
            input_size: self.input_size,
            trans_distance,
            input_index,
            output_index,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() -> Result<()> {
        let mut builder = FaceAlignmentBuilder::new("models/coor_2d106.tflite");
        let _alignment = builder
            .with_marker_nums(100)
            .with_input_size(100, 100)
            .build()?;

        Ok(())
    }
}
