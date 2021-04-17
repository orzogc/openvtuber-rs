use crate::opencv_utils::convert_mat_to_array2;
use crate::{OpenVtuberError, Point, Result};
use crate::{HEAD_POSE_INDEX, HEAD_POSE_OBJECT};
use core::convert::TryInto;
use opencv::calib3d::{decompose_projection_matrix, rodrigues, solve_pnp, SOLVEPNP_ITERATIVE};
use opencv::core::{hconcat2, no_array};
use opencv::prelude::*;

pub struct HeadPoseEstimation {
    matrix: Mat,
    obj: Mat,
}

impl HeadPoseEstimation {
    pub fn new(width: i32, height: i32) -> Result<HeadPoseEstimation> {
        let matrix = [
            [width as f32, 0., width as f32 / 2.],
            [0., width as f32, height as f32 / 2.],
            [0., 0., 1.],
        ];

        Ok(HeadPoseEstimation {
            matrix: Mat::from_slice_2d(&matrix)?,
            obj: Mat::from_slice_2d(&HEAD_POSE_OBJECT)?,
        })
    }

    pub fn get_head_pose(&self, shape: &[Point<f32>; 106]) -> Result<[f64; 3]> {
        self.solve_pnp(shape)
    }

    fn solve_pnp(&self, shape: &[Point<f32>; 106]) -> Result<[f64; 3]> {
        let shape: Vec<[f32; 2]> = HEAD_POSE_INDEX.iter().map(|i| shape[*i].into()).collect();
        let shape = Mat::from_slice_2d(shape.as_slice())?;
        let mut rotation_vec = Mat::default();
        let mut translation_vec = Mat::default();

        if !solve_pnp(
            &self.obj,
            &shape,
            &self.matrix,
            &no_array()?,
            &mut rotation_vec,
            &mut translation_vec,
            false,
            SOLVEPNP_ITERATIVE,
        )? {
            return Err(OpenVtuberError::Error("failed to solve pnp"));
        }

        let mut rotation_mat = Mat::default();
        rodrigues(&rotation_vec, &mut rotation_mat, &mut no_array()?)?;
        let mut pose_mat = Mat::default();
        hconcat2(&rotation_mat, &translation_vec, &mut pose_mat)?;
        let mut euler_angle = Mat::default();
        decompose_projection_matrix(
            &pose_mat,
            &mut Mat::default(),
            &mut Mat::default(),
            &mut Mat::default(),
            &mut no_array()?,
            &mut no_array()?,
            &mut no_array()?,
            &mut euler_angle,
        )?;

        Ok(convert_mat_to_array2(&euler_angle)?
            .as_standard_layout()
            .as_slice()
            .unwrap() // `unwrap()` is safe here.
            .try_into()?)
    }
}
