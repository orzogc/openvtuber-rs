use core::convert::TryInto;
use core::f64::consts::PI;
use ndarray::prelude::*;
use opencv::core::Point;
use opencv::highgui::{imshow, wait_key};
use opencv::imgproc::{circle, line, LINE_8, LINE_AA};
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, CAP_ANY, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH};
use openvtuber_rs::{
    FaceAlignmentBuilder, FaceDetectionBuilder, HeadPoseEstimation, Input, Points, Result,
};

fn draw_axis(image: &mut Mat, euler_angle: [f64; 3], face: Points) -> Result<()> {
    let sum = face.iter().fold([0., 0.], |s, p| [s[0] + p.x, s[1] + p.y]);
    let center = [sum[0] / face.len() as f32, sum[1] / face.len() as f32];

    let num = euler_angle
        .iter()
        .map(|a| (*a * PI / 180.).sin() as f32)
        .collect::<Vec<_>>();
    let [sin_pitch, sin_yaw, sin_roll]: [_; 3] = num.try_into().unwrap();
    let num = euler_angle
        .iter()
        .map(|a| (*a * PI / 180.).cos() as f32)
        .collect::<Vec<_>>();
    let [cos_pitch, cos_yaw, cos_roll]: [_; 3] = num.try_into().unwrap();

    let mut axis = array![
        [
            cos_yaw * cos_roll,
            cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw
        ],
        [
            -cos_yaw * sin_roll,
            cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll
        ],
        [sin_yaw, -cos_yaw * sin_pitch]
    ];
    axis *= 80.;
    axis = axis + arr1(&center);
    let axis = axis.map(|a| *a as i32);

    line(
        image,
        Point::new(center[0].round() as i32, center[1].round() as i32),
        Point::new(axis[(0, 0)], axis[(0, 1)]),
        (0., 0., 255.).into(),
        3,
        LINE_8,
        0,
    )?;
    line(
        image,
        Point::new(center[0].round() as i32, center[1].round() as i32),
        Point::new(axis[(1, 0)], axis[(1, 1)]),
        (0., 255., 0.).into(),
        3,
        LINE_8,
        0,
    )?;
    line(
        image,
        Point::new(center[0].round() as i32, center[1].round() as i32),
        Point::new(axis[(2, 0)], axis[(2, 1)]),
        (255., 0., 0.).into(),
        3,
        LINE_8,
        0,
    )?;

    Ok(())
}

fn main() -> Result<()> {
    let mut cap = VideoCapture::from_file("examples/assets/kira.gif", CAP_ANY)?;

    let mut builder = FaceDetectionBuilder::new("models/RFB-320.tflite");
    let mut detection = builder.with_conf_threshold(0.88).build()?;
    let builder = FaceAlignmentBuilder::new("models/coor_2d106.tflite");
    let mut alignment = builder.build()?;
    let estimation = HeadPoseEstimation::new(
        cap.get(CAP_PROP_FRAME_WIDTH)?.round() as i32,
        cap.get(CAP_PROP_FRAME_HEIGHT)?.round() as i32,
    )?;

    let mut frame = Mat::default();

    loop {
        if !cap.read(&mut frame)? {
            break;
        }

        let mut output = frame.clone();
        let image = Input::from_opencv_image(&frame)?;
        let boxes = detection.inference(&image)?;
        for b in &boxes {
            let pred = alignment.get_landmarks(&image, b)?;
            for p in pred.iter() {
                circle(
                    &mut output,
                    Point::new(p.x.round() as i32, p.y.round() as i32),
                    1,
                    (125., 255., 125.).into(),
                    1,
                    LINE_AA,
                    0,
                )?;
            }
            let euler_angle = estimation.get_head_pose(&pred.clone().try_into().unwrap())?;
            draw_axis(&mut output, euler_angle, pred)?;
        }

        imshow("openvtuber", &output)?;
        let key = wait_key(20)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    Ok(())
}
