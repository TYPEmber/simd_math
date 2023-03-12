#![feature(portable_simd)]
use std::{
    ops::{BitAnd, BitOr, Shl, Shr},
    simd::{LaneCount, Simd, SimdFloat, SimdPartialOrd, SimdUint, StdFloat, SupportedLaneCount},
};

fn pow_approx(x: f32, n: f32) -> f32 {
    (n * x.ln()).exp()
}

fn ln_approx(x: f32) -> f32 {
    0.0
}

fn exp(n: f32) -> f32 {
    0.0
}

/// Sign mask.
pub(crate) const SIGN_MASK: u32 = 0b1000_0000_0000_0000_0000_0000_0000_0000;

/// Exponent mask.
pub(crate) const EXPONENT_MASK: u32 = 0b0111_1111_1000_0000_0000_0000_0000_0000;

/// Mantissa mask.
pub(crate) const MANTISSA_MASK: u32 = 0b0000_0000_0111_1111_1111_1111_1111_1111;

/// Exponent mask.
pub(crate) const EXPONENT_BIAS: u32 = 127;

pub trait MathExt {
    type SimdType;
    const FOURS: Self::SimdType;
    const THREES: Self::SimdType;
    const TWOS: Self::SimdType;
    const ONES: Self::SimdType;
    const ZEROS: Self::SimdType;
    const EPSILON: Self::SimdType;

    type MaskType;
    const SIGN_MASK: Self::MaskType;
    const EXPONENT_MASK: Self::MaskType;
    const MANTISSA_BITS: Self::MaskType;
    const EXPONENT_BIAS: Self::MaskType;
    const INV_MASK: Self::MaskType;

    const A: Self::SimdType;
    const B: Self::SimdType;
    const C: Self::SimdType;
    const D: Self::SimdType;
    const E: Self::SimdType;

    const LN_2: Self::SimdType;
    const LOG2_E: Self::SimdType;

    // fn abs(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn inv(self) -> Self;
}

impl<const LANES: usize> MathExt for Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    const A: Self::SimdType = Simd::from_array([-1.741_793_9f32; LANES]);
    const B: Self::SimdType = Simd::from_array([2.821_202_6; LANES]);
    const C: Self::SimdType = Simd::from_array([-1.469_956_8; LANES]);
    const D: Self::SimdType = Simd::from_array([0.447_179_55; LANES]);
    const E: Self::SimdType = Simd::from_array([0.056_570_851; LANES]);

    const LN_2: Self::SimdType = Simd::from_array([std::f32::consts::LN_2; LANES]);
    const LOG2_E: Self::SimdType = Simd::from_array([std::f32::consts::LOG2_E; LANES]);

    type SimdType = Simd<f32, LANES>;
    const FOURS: Self::SimdType = Simd::from_array([4.0f32; LANES]);
    const THREES: Self::SimdType = Simd::from_array([3.0f32; LANES]);
    const TWOS: Self::SimdType = Simd::from_array([2.0f32; LANES]);
    const ONES: Self::SimdType = Simd::from_array([1.0f32; LANES]);
    const ZEROS: Self::SimdType = Simd::from_array([0.0f32; LANES]);
    const EPSILON: Self::SimdType = Simd::from_array([f32::EPSILON; LANES]);

    type MaskType = Simd<u32, LANES>;
    const SIGN_MASK: Simd<u32, LANES> =
        Simd::from_array([0b1000_0000_0000_0000_0000_0000_0000_0000; LANES]);
    const EXPONENT_MASK: Simd<u32, LANES> =
        Simd::from_array([0b0111_1111_1000_0000_0000_0000_0000_0000; LANES]);
    const MANTISSA_BITS: Simd<u32, LANES> = Simd::from_array([23; LANES]);
    const EXPONENT_BIAS: Simd<u32, LANES> = Simd::from_array([127; LANES]);

    const INV_MASK: Simd<u32, LANES> = Simd::from_array([0x7f00_0000; LANES]);

    // fn abs(self) -> Self {
    //     Self::from_bits(self.to_bits().bitand(Self::SIGN_MASK))
    // }

    #[inline]
    fn powf(self, n: Self) -> Self {
        // if (self - Self::ONES).abs() < Self::EPSILON {
        //     return 0.0;
        // }

        (n * self.ln()).exp()
    }

    #[inline]
    fn ln(self) -> Self {
        let x_less_than_1_mask = self.simd_lt(Self::ONES);

        // let inv_self = Self::ONES / self;
        let inv_self = self.inv();
        let mut v = x_less_than_1_mask.select(inv_self, self);

        let base2_exponent = v
            .to_bits()
            .bitand(Self::EXPONENT_MASK)
            .shr(Self::MANTISSA_BITS)
            - Self::EXPONENT_BIAS;

        let divisor = Self::from_bits(v.to_bits().bitand(Self::EXPONENT_MASK));

        v /= divisor;

        // approximate polynomial generated from maple in the post using Remez Algorithm:
        // https://en.wikipedia.org/wiki/Remez_algorithm
        // let ln_1to2_polynomial =
        //     Self::A + (Self::B + (Self::C + (Self::D - Self::E * v) * v) * v) * v;
        let ln_1to2_polynomial = (-Self::E)
            .mul_add(v, Self::D)
            .mul_add(v, Self::C)
            .mul_add(v, Self::B)
            .mul_add(v, Self::A);

        let result = base2_exponent
            .cast::<f32>()
            .mul_add(Self::LN_2, ln_1to2_polynomial);

        let minus_result = -result;

        x_less_than_1_mask.select(minus_result, result)
    }

    #[inline]
    fn exp(self) -> Self {
        // log base 2(E) == 1/ln(2)
        // x_fract + x_whole = x/ln2_recip
        // ln2*(x_fract + x_whole) = x
        let x_ln2recip = self * Self::LOG2_E;

        let x_fract = x_ln2recip.fract();
        // let x_trunc = x_ln2recip.trunc();
        let x_trunc = x_ln2recip.cast::<u32>();

        //guaranteed to be 0 < x < 1.0
        let x_fract = x_fract * Self::LN_2;
        let fract_exp = {
            let mut total = Self::ONES;

            // for i in (1..=4).rev() {
            //     total = Self::ONES + ((x_fract / Simd::splat(i as f32)) * total);
            // }
            total = total.mul_add(x_fract / Self::FOURS, Self::ONES);
            total = total.mul_add(x_fract / Self::THREES, Self::ONES);
            total = total.mul_add(x_fract / Self::TWOS, Self::ONES);
            total = total.mul_add(x_fract / Self::ONES, Self::ONES);

            total
        };

        let fract_exponent = (fract_exp
            .to_bits()
            .bitand(Self::EXPONENT_MASK)
            .shr(Self::MANTISSA_BITS)
            - Self::EXPONENT_BIAS)
            .saturating_add(x_trunc);

        // if fract_exponent < -Self::EXPONENT_BIAS.cast::<i32>() {
        //     return Self::ZEROS;
        // }

        // if fract_exponent > (Self::EXPONENT_BIAS + Self::ONES).cast::<i32>() {
        //     return Self::INFINITY;
        // }

        let without_exponent = fract_exp.to_bits().bitand(!Self::EXPONENT_MASK);
        let only_exponent = (fract_exponent + Self::EXPONENT_BIAS).shl(Self::MANTISSA_BITS);

        Self::from_bits(without_exponent.bitor(only_exponent))
    }

    fn inv(self) -> Self {
        Self::from_bits(Self::INV_MASK - self.to_bits())
    }
}

#[cfg(test)]
mod tests {
    use std::{ops::IndexMut, simd::SimdPartialOrd};

    use super::*;

    #[test]
    fn powf() {
        const N: usize = 8;
        let ones = Simd::<_, N>::splat(1f32);
        let mut v = Simd::<_, N>::splat(2f32);
        let mut n = Simd::<_, N>::splat(0.33f32);

        *n.index_mut(0) = 10.0;
        *v.index_mut(4) = 0.278;
        // let mut v = dbg!(v.inv());

        dbg!(&v.powf(n));
        dbg!(&2f32.powf(0.33f32));
        dbg!(&0.278f32.powf(0.33f32));
        dbg!(&2f32.powf(10.0f32));
    }

    #[test]
    fn exp() {
        const N: usize = 8;
        let ones = Simd::<_, N>::splat(1f32);
        let mut v = Simd::<_, N>::splat(0.0f32);

        dbg!(&v.exp());
        dbg!(&0.0f32.exp());
    }

    #[test]
    fn ln() {
        const N: usize = 8;
        let ones = Simd::<_, N>::splat(1f32);
        let mut v = Simd::<_, N>::splat(2f32);
        let mut v = dbg!(v.inv());

        *v.index_mut(2) = 1.2;

        dbg!(&v.ln());
        dbg!(1.2f32.ln());
        dbg!(0.5f32.ln());

        dbg!(v < ones);

        dbg!(&v.simd_lt(ones));

        dbg!(&v.partial_cmp(&ones));

        dbg!(pow_approx(10.0, 0.2));
        dbg!(10.0f32.powf(0.2));
    }

    #[test]
    fn shr() {
        const N: usize = 8;
        let ones = Simd::<_, N>::splat(f32::MAX);

        const EXPONENT_MASK: Simd<u32, N> =
            Simd::from_array([0b0111_1111_1000_0000_0000_0000_0000_0000; N]);

        const MANTISSA_MASK: Simd<u32, N> =
            Simd::from_array([0b0000_0000_0111_1111_1111_1111_1111_1111; N]);

        dbg!(ones.to_bits().bitand(EXPONENT_MASK).shr(MANTISSA_MASK));
        println!("{:b}", 100f32.to_bits());
        dbg!((f32::MAX.to_bits() & 0b0111_1111_1000_0000_0000_0000_0000_0000).overflowing_shr(23));
    }
}
