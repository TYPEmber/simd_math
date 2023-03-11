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
    let xs = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 255f32));
    let ns = Array1::random(LEN, ndarray_rand::rand_distr::Uniform::new(0f32, 3f32));

    c.bench_function("simd_powf", |te| {
        te.iter(|| {
            // let vs = xs
            //     .iter()
            //     .zip(ns.iter())
            //     .skip(N)
            //     .map(|(x, n)| unsafe {
            //         let xv = Simd::<_, N>::from_slice(
            //             (x as *const f32 as *const [f32; N]).as_ref().unwrap(),
            //         );
            //         let nv = Simd::<_, N>::from_slice(
            //             (n as *const f32 as *const [f32; N]).as_ref().unwrap(),
            //         );
            //         xv.powf(nv)
            //     }).collect::<Vec<_>>();

            let last = xs
                .iter()
                .copied()
                .array_chunks::<N>()
                .zip(ns.iter().copied().array_chunks::<N>())
                .map(|(x, n)| {
                    let xv = Simd::from_array(x);
                    let nv = Simd::from_array(n);
                    xv.powf(nv)
                })
                .last();
            // .collect::<Vec<_>>();

            last
        })
    });

    c.bench_function("packed_powf", |te| {
        te.iter(|| {
            use packed_simd::Simd;

            let last = xs
                .iter()
                .copied()
                .array_chunks::<N>()
                .zip(ns.iter().copied().array_chunks::<N>())
                .map(|(x, n)| {
                    let xv = Simd::from(x);
                    let nv = Simd::from(n);
                    xv.powf(nv)
                })
                .last();
            // .collect::<Vec<_>>();

            last
        })
    });

    c.bench_function("std_powf", |te| {
        te.iter(|| {
            let vs = xs.iter().zip(ns.iter()).map(|(&x, &n)| x.powf(n)).last();
            // .collect::<Vec<_>>();

            vs
        });
    });

    let xs_m = xs.mapv(micromath::F32::from);
    let ns_m = ns.mapv(micromath::F32::from);
    c.bench_function("micro_powf", |te| {
        te.iter(|| {
            let vs = xs_m
                .iter()
                .zip(ns_m.iter())
                .map(|(&x, &n)| x.powf(n))
                .last();
            // .collect::<Vec<_>>();

            vs
        });
    });
}

criterion_group!(benches, criterion_benchmark_powf,);

criterion_main!(benches);
