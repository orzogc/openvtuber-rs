use crate::opencv_utils::convert_image_mat_to_array3;
use crate::utils::convert_array3_to_image;
use crate::{OpenVtuberError, Result};
use image::imageops::{resize, FilterType};
use image::{ImageBuffer, Pixel};
use ndarray::prelude::*;
use ndarray::{Data, RawDataClone, ViewRepr};
use opencv::prelude::*;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    #[inline]
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T> From<(T, T)> for Point<T> {
    #[inline]
    fn from(p: (T, T)) -> Self {
        Self { x: p.0, y: p.1 }
    }
}

impl<T> From<Point<T>> for (T, T) {
    #[inline]
    fn from(p: Point<T>) -> Self {
        (p.x, p.y)
    }
}

impl<T: Copy> From<[T; 2]> for Point<T> {
    #[inline]
    fn from(p: [T; 2]) -> Self {
        Self { x: p[0], y: p[1] }
    }
}

impl<T> From<Point<T>> for [T; 2] {
    #[inline]
    fn from(p: Point<T>) -> Self {
        [p.x, p.y]
    }
}

impl<T> From<Point<T>> for Vec<T> {
    #[inline]
    fn from(p: Point<T>) -> Self {
        vec![p.x, p.y]
    }
}

impl<T> From<Point<T>> for Array1<T> {
    #[inline]
    fn from(p: Point<T>) -> Self {
        array![p.x, p.y]
    }
}

impl<T: core::ops::Add<Output = T>> core::ops::Add for Point<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: core::ops::Add<Output = T> + Copy> core::ops::Add<T> for Point<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self {
            x: self.x + rhs,
            y: self.y + rhs,
        }
    }
}

impl<T: core::ops::AddAssign> core::ops::AddAssign for Point<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: core::ops::AddAssign + Copy> core::ops::AddAssign<T> for Point<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.x += rhs;
        self.y += rhs;
    }
}

impl<T: core::ops::Sub<Output = T>> core::ops::Sub for Point<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: core::ops::Sub<Output = T> + Copy> core::ops::Sub<T> for Point<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self {
            x: self.x - rhs,
            y: self.y - rhs,
        }
    }
}

impl<T: core::ops::SubAssign> core::ops::SubAssign for Point<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: core::ops::SubAssign + Copy> core::ops::SubAssign<T> for Point<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.x -= rhs;
        self.y -= rhs;
    }
}

impl<T: core::ops::Mul<Output = T> + Copy> core::ops::Mul<T> for Point<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: core::ops::MulAssign + Copy> core::ops::MulAssign<T> for Point<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: core::ops::Div<Output = T> + Copy> core::ops::Div<T> for Point<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: core::ops::DivAssign + Copy> core::ops::DivAssign<T> for Point<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T: core::ops::Rem<Output = T> + Copy> core::ops::Rem<T> for Point<T> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: T) -> Self::Output {
        Self {
            x: self.x % rhs,
            y: self.y % rhs,
        }
    }
}

impl<T: core::ops::RemAssign + Copy> core::ops::RemAssign<T> for Point<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: T) {
        self.x %= rhs;
        self.y %= rhs;
    }
}

impl<T: core::ops::Neg<Output = T>> core::ops::Neg for Point<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Rectangle<T> {
    pub x1: T,
    pub y1: T,
    pub x2: T,
    pub y2: T,
}

impl<T> Rectangle<T> {
    #[inline]
    pub fn new(x1: T, y1: T, x2: T, y2: T) -> Self {
        Self { x1, y1, x2, y2 }
    }

    #[inline]
    pub fn width(&self) -> T
    where
        T: core::ops::Sub<Output = T> + Copy,
    {
        self.x2 - self.x1
    }

    #[inline]
    pub fn height(&self) -> T
    where
        T: core::ops::Sub<Output = T> + Copy,
    {
        self.y2 - self.y1
    }
}

impl<T> From<(T, T, T, T)> for Rectangle<T> {
    #[inline]
    fn from(r: (T, T, T, T)) -> Self {
        Self {
            x1: r.0,
            y1: r.1,
            x2: r.2,
            y2: r.3,
        }
    }
}

impl<T> From<Rectangle<T>> for (T, T, T, T) {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        (r.x1, r.y1, r.x2, r.y2)
    }
}

impl<T> From<(Point<T>, Point<T>)> for Rectangle<T> {
    #[inline]
    fn from((p1, p2): (Point<T>, Point<T>)) -> Self {
        Self {
            x1: p1.x,
            y1: p1.y,
            x2: p2.x,
            y2: p2.y,
        }
    }
}

impl<T> From<Rectangle<T>> for (Point<T>, Point<T>) {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        ((r.x1, r.y1).into(), (r.x2, r.y2).into())
    }
}

impl<T: Copy> From<[T; 4]> for Rectangle<T> {
    #[inline]
    fn from(r: [T; 4]) -> Self {
        Self {
            x1: r[0],
            y1: r[1],
            x2: r[2],
            y2: r[3],
        }
    }
}

impl<T> From<Rectangle<T>> for [T; 4] {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        [r.x1, r.y1, r.x2, r.y2]
    }
}

impl<T: Copy> From<[Point<T>; 2]> for Rectangle<T> {
    #[inline]
    fn from(r: [Point<T>; 2]) -> Self {
        Self {
            x1: r[0].x,
            y1: r[0].y,
            x2: r[1].x,
            y2: r[1].y,
        }
    }
}

impl<T> From<Rectangle<T>> for [Point<T>; 2] {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        [(r.x1, r.y1).into(), (r.x2, r.y2).into()]
    }
}

impl<T> From<Rectangle<T>> for Vec<T> {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        vec![r.x1, r.y1, r.x2, r.y2]
    }
}

impl<T> From<Rectangle<T>> for Vec<Point<T>> {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        vec![(r.x1, r.y1).into(), (r.x2, r.y2).into()]
    }
}

impl<T> From<Rectangle<T>> for Array1<T> {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        array![r.x1, r.y1, r.x2, r.y2]
    }
}

impl<T> From<Rectangle<T>> for Array2<T> {
    #[inline]
    fn from(r: Rectangle<T>) -> Self {
        array![[r.x1, r.y1], [r.x2, r.y2]]
    }
}

impl<T: core::ops::Add<Output = T> + Copy> core::ops::Add<T> for Rectangle<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self {
            x1: self.x1 + rhs,
            y1: self.y1 + rhs,
            x2: self.x2 + rhs,
            y2: self.y2 + rhs,
        }
    }
}

impl<T: core::ops::Sub<Output = T> + Copy> core::ops::Sub<T> for Rectangle<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self {
            x1: self.x1 - rhs,
            y1: self.y1 - rhs,
            x2: self.x2 - rhs,
            y2: self.y2 - rhs,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Points(pub Vec<Point<f32>>);

impl Points {
    #[inline]
    pub fn new(vec: Vec<Point<f32>>) -> Self {
        Self(vec)
    }
}

impl From<Vec<Point<f32>>> for Points {
    #[inline]
    fn from(p: Vec<Point<f32>>) -> Self {
        Self(p)
    }
}

impl From<Points> for Vec<Point<f32>> {
    #[inline]
    fn from(p: Points) -> Self {
        p.0
    }
}

impl<const N: usize> From<[Point<f32>; N]> for Points {
    #[inline]
    fn from(p: [Point<f32>; N]) -> Self {
        Self(p.into())
    }
}

impl<const N: usize> core::convert::TryFrom<Points> for [Point<f32>; N] {
    type Error = OpenVtuberError;

    #[inline]
    fn try_from(p: Points) -> Result<Self> {
        use core::convert::TryInto;

        p.0.try_into().or(Err(OpenVtuberError::Error(
            "failed to convert `Points` to `[Point<f32>; N]`",
        )))
    }
}

impl<'a, const N: usize> core::convert::TryFrom<&'a Points> for &'a [Point<f32>; N] {
    type Error = OpenVtuberError;

    #[inline]
    fn try_from(p: &'a Points) -> Result<Self> {
        use core::convert::TryInto;

        Ok(p.as_slice().try_into()?)
    }
}

impl<'a, const N: usize> core::convert::TryFrom<&'a mut Points> for &'a mut [Point<f32>; N] {
    type Error = OpenVtuberError;

    #[inline]
    fn try_from(p: &'a mut Points) -> Result<Self> {
        use core::convert::TryInto;

        Ok(p.as_mut_slice().try_into()?)
    }
}

impl From<&[Point<f32>]> for Points {
    #[inline]
    fn from(p: &[Point<f32>]) -> Self {
        p.to_vec().into()
    }
}

impl<'a> From<&'a Points> for &'a [Point<f32>] {
    #[inline]
    fn from(p: &'a Points) -> Self {
        p.as_slice()
    }
}

impl<'a> From<&'a mut Points> for &'a mut [Point<f32>] {
    #[inline]
    fn from(p: &'a mut Points) -> Self {
        p.as_mut_slice()
    }
}

impl core::ops::Deref for Points {
    type Target = Vec<Point<f32>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Points {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl core::convert::AsRef<Vec<Point<f32>>> for Points {
    #[inline]
    fn as_ref(&self) -> &Vec<Point<f32>> {
        &self.0
    }
}

impl core::convert::AsMut<Vec<Point<f32>>> for Points {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<Point<f32>> {
        &mut self.0
    }
}

impl core::iter::IntoIterator for Points {
    type Item = Point<f32>;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> core::iter::IntoIterator for &'a Points {
    type Item = &'a Point<f32>;

    type IntoIter = core::slice::Iter<'a, Point<f32>>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).iter()
    }
}

impl<'a> core::iter::IntoIterator for &'a mut Points {
    type Item = &'a mut Point<f32>;

    type IntoIter = core::slice::IterMut<'a, Point<f32>>;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.0).iter_mut()
    }
}

impl core::ops::Index<usize> for Points {
    type Output = Point<f32>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<usize> for Points {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rectangles(pub Vec<Rectangle<f32>>);

impl Rectangles {
    #[inline]
    pub fn new(vec: Vec<Rectangle<f32>>) -> Self {
        Self(vec)
    }
}

impl From<Vec<Rectangle<f32>>> for Rectangles {
    #[inline]
    fn from(r: Vec<Rectangle<f32>>) -> Self {
        Self(r)
    }
}

impl From<Rectangles> for Vec<Rectangle<f32>> {
    #[inline]
    fn from(r: Rectangles) -> Self {
        r.0
    }
}

impl<const N: usize> From<[Rectangle<f32>; N]> for Rectangles {
    #[inline]
    fn from(r: [Rectangle<f32>; N]) -> Self {
        Self(r.into())
    }
}

impl<const N: usize> core::convert::TryFrom<Rectangles> for [Rectangle<f32>; N] {
    type Error = OpenVtuberError;

    #[inline]
    fn try_from(r: Rectangles) -> Result<Self> {
        use core::convert::TryInto;

        r.0.try_into().or(Err(OpenVtuberError::Error(
            "failed to convert `Rectangles` to `[Rectangle<f32>; N]`",
        )))
    }
}

impl<'a, const N: usize> core::convert::TryFrom<&'a Rectangles> for &'a [Rectangle<f32>; N] {
    type Error = OpenVtuberError;

    #[inline]
    fn try_from(r: &'a Rectangles) -> Result<Self> {
        use core::convert::TryInto;

        Ok(r.as_slice().try_into()?)
    }
}

impl<'a, const N: usize> core::convert::TryFrom<&'a mut Rectangles>
    for &'a mut [Rectangle<f32>; N]
{
    type Error = OpenVtuberError;

    #[inline]
    fn try_from(r: &'a mut Rectangles) -> Result<Self> {
        use core::convert::TryInto;

        Ok(r.as_mut_slice().try_into()?)
    }
}

impl From<&[Rectangle<f32>]> for Rectangles {
    #[inline]
    fn from(r: &[Rectangle<f32>]) -> Self {
        r.to_vec().into()
    }
}

impl<'a> From<&'a Rectangles> for &'a [Rectangle<f32>] {
    #[inline]
    fn from(r: &'a Rectangles) -> Self {
        r.as_slice()
    }
}

impl<'a> From<&'a mut Rectangles> for &'a mut [Rectangle<f32>] {
    #[inline]
    fn from(r: &'a mut Rectangles) -> Self {
        r.as_mut_slice()
    }
}

impl core::ops::Deref for Rectangles {
    type Target = Vec<Rectangle<f32>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Rectangles {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl core::convert::AsRef<Vec<Rectangle<f32>>> for Rectangles {
    #[inline]
    fn as_ref(&self) -> &Vec<Rectangle<f32>> {
        &self.0
    }
}

impl core::convert::AsMut<Vec<Rectangle<f32>>> for Rectangles {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<Rectangle<f32>> {
        &mut self.0
    }
}

impl core::iter::IntoIterator for Rectangles {
    type Item = Rectangle<f32>;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> core::iter::IntoIterator for &'a Rectangles {
    type Item = &'a Rectangle<f32>;

    type IntoIter = core::slice::Iter<'a, Rectangle<f32>>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).iter()
    }
}

impl<'a> core::iter::IntoIterator for &'a mut Rectangles {
    type Item = &'a mut Rectangle<f32>;

    type IntoIter = core::slice::IterMut<'a, Rectangle<f32>>;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.0).iter_mut()
    }
}

impl core::ops::Index<usize> for Rectangles {
    type Output = Rectangle<f32>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<usize> for Rectangles {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Input<S>(pub ArrayBase<S, Ix3>)
where
    S: Data<Elem = u8> + RawDataClone;

impl<S> Input<S>
where
    S: Data<Elem = u8> + RawDataClone,
{
    #[inline]
    pub fn width(&self) -> usize {
        self.shape()[1]
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.shape()[0]
    }

    #[inline]
    pub fn as_image<P>(&self) -> Result<ImageBuffer<P, &[u8]>>
    where
        P: 'static + Pixel<Subpixel = u8>,
    {
        convert_array3_to_image(self.view())
    }

    pub fn resize<P>(
        &self,
        width: usize,
        height: usize,
        filter: FilterType,
    ) -> Result<ImageBuffer<P, Vec<u8>>>
    where
        P: 'static + Pixel<Subpixel = u8>,
    {
        let image = self.as_standard_layout();
        let image = convert_array3_to_image(image.view())?;

        Ok(resize(&image, width as u32, height as u32, filter))
    }
}

impl<'a> Input<ViewRepr<&'a u8>> {
    #[inline]
    pub fn from_opencv_image(image: &'a Mat) -> Result<Self> {
        Ok(convert_image_mat_to_array3(image)?.into())
    }
}

impl<S> From<ArrayBase<S, Ix3>> for Input<S>
where
    S: Data<Elem = u8> + RawDataClone,
{
    #[inline]
    fn from(i: ArrayBase<S, Ix3>) -> Self {
        Self(i)
    }
}

impl<S> From<Input<S>> for ArrayBase<S, Ix3>
where
    S: Data<Elem = u8> + RawDataClone,
{
    #[inline]
    fn from(i: Input<S>) -> Self {
        i.0
    }
}

impl<'a, S> From<&'a Input<S>> for ArrayView3<'a, u8>
where
    S: Data<Elem = u8> + RawDataClone,
{
    #[inline]
    fn from(i: &'a Input<S>) -> Self {
        i.0.view()
    }
}

impl<S> core::ops::Deref for Input<S>
where
    S: Data<Elem = u8> + RawDataClone,
{
    type Target = ArrayBase<S, Ix3>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S> core::ops::DerefMut for Input<S>
where
    S: Data<Elem = u8> + RawDataClone,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<S> core::convert::AsRef<ArrayBase<S, Ix3>> for Input<S>
where
    S: Data<Elem = u8> + RawDataClone,
{
    #[inline]
    fn as_ref(&self) -> &ArrayBase<S, Ix3> {
        &self.0
    }
}

impl<S> core::convert::AsMut<ArrayBase<S, Ix3>> for Input<S>
where
    S: Data<Elem = u8> + RawDataClone,
{
    #[inline]
    fn as_mut(&mut self) -> &mut ArrayBase<S, Ix3> {
        &mut self.0
    }
}
