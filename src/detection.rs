use crate::utils::{
    build_interpreter, clip_f32, convert_image_to_array3, non_max_suppression, normalize_min_max,
};
use crate::{Input, Rectangle, Rectangles, Result};
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};
use ndarray::prelude::*;
use ndarray::{concatenate, stack, Data, RawDataClone};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::Interpreter;

pub struct FaceDetection<'a> {
    interpreter: Interpreter<'a, BuiltinOpResolver>,
    input_index: i32,
    boxes_index: i32,
    scores_index: i32,
    anchors_xy: Array2<f32>,
    anchors_wh: Array2<f32>,
    input_size: [i32; 2],
    conf_threshold: f32,
    center_variance: f32,
    size_variance: f32,
    nms_iou_threshold: f32,
}

impl<'a> FaceDetection<'a> {
    pub fn inference<S>(&mut self, image: &Input<S>) -> Result<Rectangles>
    where
        S: Data<Elem = u8> + RawDataClone,
    {
        let width = image.width() as f32;
        let height = image.height() as f32;
        let resized_image: ImageBuffer<Rgb<_>, _> = image.resize(
            self.input_size[0] as usize,
            self.input_size[1] as usize,
            FilterType::Nearest,
        )?;
        let resized_image = convert_image_to_array3(&resized_image)?;

        let input = self.pre_processing(resized_image)?;

        self.set_input(input)?;
        self.interpreter.invoke()?;

        let boxes = self.get_boxes()?;
        let scores = self.get_scores()?;

        let mut boxes = self.post_processing(boxes, scores)?;
        for b in boxes.iter_mut() {
            b.x1 *= width;
            b.y1 *= height;
            b.x2 *= width;
            b.y2 *= height;
        }

        Ok(boxes)
    }

    fn pre_processing(&self, image: ArrayView3<u8>) -> Result<Array4<f32>> {
        let image = image.map(|i| *i as f32);
        let norm = normalize_min_max(image, -1., 1.)?;

        Ok(norm.insert_axis(Axis(0)))
    }

    fn post_processing(&self, boxes: Array2<f32>, scores: Array2<f32>) -> Result<Rectangles> {
        let boxes = self.decode_regression(boxes);
        let scores = scores.index_axis(Axis(1), 1);

        let conf_mask = scores.map(|s| *s > self.conf_threshold);
        let boxes: Vec<Rectangle<f32>> = boxes
            .outer_iter()
            .zip(&conf_mask)
            .filter_map(|(bo, b)| {
                if *b {
                    // `unwrap()` is safe here.
                    Some(bo.as_standard_layout().as_slice().unwrap().into())
                } else {
                    None
                }
            })
            .collect();

        let scores = scores
            .iter()
            .zip(&conf_mask)
            .filter_map(|(s, b)| if *b { Some(*s) } else { None })
            .collect::<Vec<_>>();

        Ok(non_max_suppression(
            boxes.into(),
            scores,
            self.nms_iou_threshold,
        ))
    }

    fn decode_regression(&self, reg: Array2<f32>) -> Array2<f32> {
        let (center_xy, center_wh) = reg.view().split_at(Axis(1), 2);
        let center_xy = center_xy.to_owned() * self.center_variance * self.anchors_wh.view()
            + self.anchors_xy.view();
        let center_wh = (center_wh.to_owned() * self.size_variance).mapv_into(f32::exp)
            * self.anchors_wh.view()
            / 2.;

        let start_xy = center_xy.clone() - center_wh.view();
        let end_xy = center_xy + center_wh;

        let boxes = concatenate![Axis(1), start_xy, end_xy];
        clip_f32(boxes, 0., 1.)
    }

    fn set_input(&mut self, input: Array4<f32>) -> Result<()> {
        let input_tensor: &mut [f32] = self.interpreter.tensor_data_mut(self.input_index)?;
        input_tensor
            .iter_mut()
            .zip(&input)
            .for_each(|(it, i)| *it = *i);

        Ok(())
    }

    fn get_boxes(&self) -> Result<Array2<f32>> {
        let boxes: &[f32] = self.interpreter.tensor_data(self.boxes_index)?;
        Ok(ArrayBase::from_shape_vec(
            (boxes.len() / 4, 4),
            boxes.to_vec(),
        )?)
    }

    fn get_scores(&self) -> Result<Array2<f32>> {
        let scores: &[f32] = self.interpreter.tensor_data(self.scores_index)?;
        Ok(ArrayBase::from_shape_vec(
            (scores.len() / 2, 2),
            scores.to_vec(),
        )?)
    }
}

#[derive(Clone, Debug)]
pub struct FaceDetectionBuilder {
    filepath: String,
    input_size: [i32; 2],
    conf_threshold: f32,
    center_variance: f32,
    size_variance: f32,
    nms_iou_threshold: f32,
}

impl FaceDetectionBuilder {
    #[inline]
    pub fn new(filepath: impl Into<String>) -> Self {
        FaceDetectionBuilder {
            filepath: filepath.into(),
            input_size: [320, 240],
            conf_threshold: 0.6,
            center_variance: 0.1,
            size_variance: 0.2,
            nms_iou_threshold: 0.3,
        }
    }

    #[inline]
    pub fn with_input_size(&mut self, width: i32, height: i32) -> &mut Self {
        self.input_size = [width, height];
        self
    }

    #[inline]
    pub fn with_conf_threshold(&mut self, conf_threshold: f32) -> &mut Self {
        self.conf_threshold = conf_threshold;
        self
    }

    #[inline]
    pub fn with_center_variance(&mut self, center_variance: f32) -> &mut Self {
        self.center_variance = center_variance;
        self
    }

    #[inline]
    pub fn with_size_variance(&mut self, size_variance: f32) -> &mut Self {
        self.size_variance = size_variance;
        self
    }

    #[inline]
    pub fn with_nms_iou_threshold(&mut self, nms_iou_threshold: f32) -> &mut Self {
        self.nms_iou_threshold = nms_iou_threshold;
        self
    }

    pub fn build(&self) -> Result<FaceDetection> {
        let (anchors_xy, anchors_wh) = self.generate_anchors()?;

        let interpreter = build_interpreter(self.filepath.as_str())?;
        let input_index = interpreter.inputs()[0];
        let boxes_index = interpreter.outputs()[0];
        let scores_index = interpreter.outputs()[1];

        Ok(FaceDetection {
            interpreter,
            input_index,
            boxes_index,
            scores_index,
            anchors_xy,
            anchors_wh,
            input_size: self.input_size,
            conf_threshold: self.conf_threshold,
            center_variance: self.center_variance,
            size_variance: self.size_variance,
            nms_iou_threshold: self.nms_iou_threshold,
        })
    }

    fn generate_anchors(&self) -> Result<(Array2<f32>, Array2<f32>)> {
        const FEATURE_MAPS: [[usize; 2]; 4] = [[40, 30], [20, 15], [10, 8], [5, 4]];
        let min_boxes = vec![
            array![10., 16., 24.],
            array![32., 48.],
            array![64., 96.],
            array![128., 192., 256.],
        ];
        let input_size = array![[self.input_size[0] as f32], [self.input_size[1] as f32]];
        let mut anchors = vec![];

        for (feature_map_w_h, min_box) in FEATURE_MAPS.iter().zip(min_boxes) {
            let mut len = min_box.len();
            let mut wh_grid = (min_box / input_size.view()).reversed_axes();
            wh_grid = (0..(feature_map_w_h.iter().product::<usize>() - 1))
                .fold(wh_grid.clone(), |g, _| concatenate![Axis(0), g, wh_grid]);

            let x_grid = Array::range(0., feature_map_w_h[0] as f32, 1.);
            let mut x_grid = (0..(feature_map_w_h[1] - 1))
                .fold(x_grid.clone(), |g, _| concatenate![Axis(0), g, x_grid])
                .into_shape((feature_map_w_h[1], feature_map_w_h[0]))?;
            let y_grid = Array::range(0., feature_map_w_h[1] as f32, 1.);
            let mut y_grid = (0..(feature_map_w_h[0] - 1))
                .fold(y_grid.clone(), |g, _| concatenate![Axis(0), g, y_grid])
                .into_shape((feature_map_w_h[0], feature_map_w_h[1]))?
                .reversed_axes();
            x_grid = (x_grid + 0.5) / array![[feature_map_w_h[0] as f32]];
            y_grid = (y_grid + 0.5) / array![[feature_map_w_h[1] as f32]];

            let mut xy_grid = stack![Axis(2), x_grid, y_grid];
            xy_grid =
                (0..(len - 1)).fold(xy_grid.clone(), |g, _| concatenate![Axis(2), g, xy_grid]);
            len = xy_grid.len();
            let xy_grid = xy_grid.into_shape((len / 2, 2))?;

            let prior = concatenate![Axis(1), xy_grid, wh_grid];
            anchors.push(prior);
        }

        let mut anchor = anchors.remove(0);
        anchor = anchors
            .into_iter()
            .fold(anchor, |a, an| concatenate![Axis(0), a, an]);
        let (anchors_xy, anchors_wh) = clip_f32(anchor.view_mut(), 0., 1.).split_at(Axis(1), 2);

        Ok((anchors_xy.to_owned(), anchors_wh.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_anchors() -> Result<()> {
        let builder = FaceDetectionBuilder::new("");
        let (anchors_xy, anchors_wh) = builder.generate_anchors()?;
        assert_eq!(anchors_xy.shape(), &[4420, 2]);
        assert_eq!(anchors_wh.shape(), &[4420, 2]);

        Ok(())
    }

    #[test]
    fn test_builder() -> Result<()> {
        let mut builder = FaceDetectionBuilder::new("models/RFB-320.tflite");
        let _detection = builder
            .with_input_size(100, 100)
            .with_conf_threshold(0.88)
            .with_center_variance(0.2)
            .with_size_variance(0.1)
            .with_nms_iou_threshold(0.2)
            .build()?;

        Ok(())
    }
}
