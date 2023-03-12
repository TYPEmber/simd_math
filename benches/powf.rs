#![feature(iter_array_chunks)]
#![feature(portable_simd)]

use std::simd::Simd;

use criterion::{criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use simd_math::MathExt;

fn criterion_benchmark_powf(c: &mut Criterion) {
    const LEN: usize = 1000 * 6000;
    const N: usize = 8;
    let xs = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 255f32)).to_vec();
    let ns = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 3f32)).to_vec();

    c.bench_function("simd_powf", |te| {
        te.iter(|| {
            let vs = xs
                .iter()
                .copied()
                .array_chunks::<N>()
                .zip(ns.iter().copied().array_chunks::<N>())
                .map(|(x, n)| {
                    let xv = Simd::from_array(x);
                    let nv = Simd::from_array(n);
                    xv.powf(nv)
                })
                .collect::<Vec<_>>();

            vs
        })
    });

    c.bench_function("packed_powf", |te| {
        te.iter(|| {
            use packed_simd::Simd;

            let vs = xs
                .iter()
                .copied()
                .array_chunks::<N>()
                .zip(ns.iter().copied().array_chunks::<N>())
                .map(|(x, n)| {
                    let xv = Simd::from(x);
                    let nv = Simd::from(n);
                    xv.powf(nv)
                })
                .collect::<Vec<_>>();

            vs
        })
    });

    c.bench_function("sleef_powf", |te| {
        te.iter(|| {
            let vs = xs
                .iter()
                .copied()
                .array_chunks::<N>()
                .zip(ns.iter().copied().array_chunks::<N>())
                .map(|(x, n)| {
                    let xv = Simd::from_array(x);
                    let nv = Simd::from_array(n);
                    sleef::f32x::pow_fast(xv, nv)
                })
                .collect::<Vec<_>>();

            vs
        })
    });

    c.bench_function("std_powf", |te| {
        te.iter(|| {
            let vs = xs
                .iter()
                .zip(ns.iter())
                .map(|(&x, &n)| x.powf(n))
                .collect::<Vec<_>>();

            vs
        });
    });

    let xs = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 255f32));
    let ns = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 3f32));
    let xs_m = xs.mapv(micromath::F32::from).to_vec();
    let ns_m = ns.mapv(micromath::F32::from).to_vec();
    c.bench_function("micro_powf", |te| {
        te.iter(|| {
            let vs = xs_m
                .iter()
                .zip(ns_m.iter())
                .map(|(&x, &n)| x.powf(n))
                .collect::<Vec<_>>();

            vs
        });
    });
}

criterion_group!(benches, criterion_benchmark_powf,);

criterion_main!(benches);
