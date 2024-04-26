use core::arch::global_asm;
#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;

#[cfg(target_feature = "neon")]
use common::types::ScoreType;
use half::f16;
use num_traits::Float;

use crate::data_types::vectors::{DenseVector, VectorElementTypeHalf};
use crate::spaces::tools::is_length_zero_or_normalized;
#[cfg(target_feature = "neon")]

global_asm!(include_str!("arm.s"));

extern "C" {
    fn dotProduct_half_4x4(v1: *const f16, v2: *const f16, n: i32) -> f32;
    fn euclideanDist_half_4x4(v1: *const f16, v2: *const f16, n: i32) -> f32;
    fn manhattanDist_half_4x4(v1: *const f16, v2: *const f16, n: i32) -> f32;
}

#[cfg(target_feature = "neon")]
pub(crate) unsafe fn euclid_similarity_neon(
    v1: &[VectorElementTypeHalf],
    v2: &[VectorElementTypeHalf],
) -> ScoreType {
    let n = v1.len();
    -euclideanDist_half_4x4(v1.as_ptr(), v2.as_ptr(), n.try_into().unwrap())
}

#[cfg(target_feature = "neon")]
pub(crate) unsafe fn manhattan_similarity_neon(
    v1: &[VectorElementTypeHalf],
    v2: &[VectorElementTypeHalf],
) -> ScoreType {
    let n = v1.len();
    -manhattanDist_half_4x4(v1.as_ptr(), v2.as_ptr(), n.try_into().unwrap())
}

// #[cfg(target_feature = "neon")]
// pub(crate) unsafe fn cosine_preprocess_neon(vector: &[VectorElementTypeHalf]) -> &[VectorElementTypeHalf] {
//     let n = vector.len();

//     let length = manhattanDist_half_4x4(vector.as_ptr(), vector.as_ptr(), n.try_into().unwrap());
//     let length = length.sqrt();
//     vector.into_iter().map(|x| x / length).collect()
// }

#[cfg(target_feature = "neon")]
pub(crate) unsafe fn dot_similarity_neon(
    v1: &[VectorElementTypeHalf],
    v2: &[VectorElementTypeHalf],
) -> ScoreType {
    let n: i32 = v1.len().try_into().unwrap();
    dotProduct_half_4x4(v1.as_ptr(), v2.as_ptr(), n)
}

#[cfg(test)]
mod tests {
    #[cfg(target_feature = "neon")]
    #[test]
    fn test_spaces_neon() {
        use super::*;
        use crate::spaces::metric_f16::simple::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            let v1: Vec<f16> = vec![
                f16::from_f32(1.),
                f16::from_f32(2.),
                f16::from_f32(3.),
                f16::from_f32(4.),
                f16::from_f32(5.),
                f16::from_f32(6.),
                f16::from_f32(7.),
                f16::from_f32(8.),
                f16::from_f32(9.),
                f16::from_f32(10.),
                f16::from_f32(11.),
                f16::from_f32(12.),
                f16::from_f32(13.),
                f16::from_f32(14.),
                f16::from_f32(15.),
                f16::from_f32(16.),
                f16::from_f32(17.),
                f16::from_f32(18.),
                f16::from_f32(19.),
                f16::from_f32(20.),
                f16::from_f32(21.),
                f16::from_f32(22.),
                f16::from_f32(1.),
                f16::from_f32(2.),
                f16::from_f32(3.),
                f16::from_f32(4.),
                f16::from_f32(5.),
                f16::from_f32(6.),
                f16::from_f32(7.),
                f16::from_f32(8.),
                f16::from_f32(9.),
                f16::from_f32(10.),
                f16::from_f32(11.),
                f16::from_f32(12.),
                f16::from_f32(13.),
                f16::from_f32(14.),
                f16::from_f32(15.),
                f16::from_f32(16.),
                f16::from_f32(17.),
                f16::from_f32(18.),
                f16::from_f32(19.),
                f16::from_f32(20.),
                f16::from_f32(21.),
                f16::from_f32(22.),
            ];
            let v2: Vec<f16> = vec![
                f16::from_f32(2.),
                f16::from_f32(3.),
                f16::from_f32(4.),
                f16::from_f32(5.),
                f16::from_f32(6.),
                f16::from_f32(7.),
                f16::from_f32(8.),
                f16::from_f32(9.),
                f16::from_f32(10.),
                f16::from_f32(11.),
                f16::from_f32(12.),
                f16::from_f32(13.),
                f16::from_f32(14.),
                f16::from_f32(15.),
                f16::from_f32(16.),
                f16::from_f32(17.),
                f16::from_f32(18.),
                f16::from_f32(19.),
                f16::from_f32(20.),
                f16::from_f32(21.),
                f16::from_f32(22.),
                f16::from_f32(23.),
                f16::from_f32(2.),
                f16::from_f32(3.),
                f16::from_f32(4.),
                f16::from_f32(5.),
                f16::from_f32(6.),
                f16::from_f32(7.),
                f16::from_f32(8.),
                f16::from_f32(9.),
                f16::from_f32(10.),
                f16::from_f32(11.),
                f16::from_f32(12.),
                f16::from_f32(13.),
                f16::from_f32(14.),
                f16::from_f32(15.),
                f16::from_f32(16.),
                f16::from_f32(17.),
                f16::from_f32(18.),
                f16::from_f32(19.),
                f16::from_f32(20.),
                f16::from_f32(21.),
                f16::from_f32(22.),
                f16::from_f32(23.),
            ];

            let euclid_simd = unsafe { euclid_similarity_neon(&v1, &v2) };
            let euclid = euclid_similarity_half(&v1, &v2);
            assert_eq!(euclid_simd, euclid);

            let manhattan_simd = unsafe { manhattan_similarity_neon(&v1, &v2) };
            let manhattan = manhattan_similarity_half(&v1, &v2);
            assert_eq!(manhattan_simd, manhattan);

            let dot_simd = unsafe { dot_similarity_neon(&v1, &v2) };
            let dot = dot_similarity_half(&v1, &v2);
            assert_eq!(dot_simd, dot);

            // let cosine_simd = unsafe { cosine_preprocess_neon(v1.clone()) };
            // let cosine = cosine_preprocess(v1);
            // assert_eq!(cosine_simd, cosine);
        } else {
            println!("neon test skipped");
        }
    }
}
