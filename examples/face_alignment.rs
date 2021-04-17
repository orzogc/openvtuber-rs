use opencv::core::Point;
use opencv::highgui::{imshow, wait_key};
use opencv::imgproc::{circle, LINE_AA};
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, CAP_ANY};
use openvtuber_rs::{FaceAlignmentBuilder, FaceDetectionBuilder, Input, Result};

fn main() -> Result<()> {
    let mut builder = FaceDetectionBuilder::new("models/RFB-320.tflite");
    let mut detection = builder.with_conf_threshold(0.88).build()?;
    let builder = FaceAlignmentBuilder::new("models/coor_2d106.tflite");
    let mut alignment = builder.build()?;

    let mut cap = VideoCapture::from_file("examples/assets/kira.gif", CAP_ANY)?;
    let mut frame = Mat::default();

    loop {
        if !cap.read(&mut frame)? {
            break;
        }

        let mut output = frame.clone();
        let image = Input::from_opencv_image(&frame)?;
        let boxes = detection.inference(&image)?;
        for b in &boxes {
            for p in alignment.get_landmarks(&image, b)? {
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
        }

        imshow("openvtuber", &output)?;
        let key = wait_key(20)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    Ok(())
}
