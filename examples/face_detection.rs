use opencv::core::Rect;
use opencv::highgui::{imshow, wait_key};
use opencv::imgproc::{rectangle, LINE_8};
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, CAP_ANY};
use openvtuber_rs::{FaceDetectionBuilder, Input, Result};

fn main() -> Result<()> {
    let mut builder = FaceDetectionBuilder::new("models/RFB-320.tflite");
    let mut detection = builder.with_conf_threshold(0.88).build()?;

    let mut cap = VideoCapture::from_file("examples/assets/kira.gif", CAP_ANY)?;
    let mut frame = Mat::default();

    loop {
        if !cap.read(&mut frame)? {
            break;
        }

        let image = Input::from_opencv_image(&frame)?;
        let boxes = detection.inference(&image)?;
        for b in boxes {
            rectangle(
                &mut frame,
                Rect::new(
                    b.x1.round() as i32,
                    b.y1.round() as i32,
                    b.width().round() as i32,
                    b.height().round() as i32,
                ),
                (2., 255., 0.).into(),
                1,
                LINE_8,
                0,
            )?;
        }

        imshow("openvtuber", &frame)?;
        let key = wait_key(20)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    Ok(())
}
