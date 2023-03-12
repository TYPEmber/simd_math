#![feature(iter_array_chunks)]
#![feature(portable_simd)]

use std::simd::{Simd, SimdFloat};

use criterion::{criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ndarray_rand::RandomExt;

fn criterion_benchmark_powf(c: &mut Criterion) {
    const LEN: usize = 1000 * 6000;
    const N: usize = 8;
    let xs = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 255f32));
    let ns = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 3f32));

    let FIVES = Simd::<_, N>::splat(5.0);
    let ONES = Simd::<_, N>::splat(1.0);

    c.bench_function("mul_and_add", |te| {
        te.iter(|| {
            xs.iter()
                .map(|i| Simd::splat(*i) * FIVES + ONES)
                .last()
                .unwrap()
        });
    });

    use std::simd::StdFloat;
    c.bench_function("fma", |te| {
        te.iter(|| {
            xs.iter()
                .map(|i| Simd::splat(*i).mul_add(FIVES, ONES))
                .last()
                .unwrap()
        });
    });
}

criterion_group!(benches, criterion_benchmark_powf,);

criterion_main!(benches);
