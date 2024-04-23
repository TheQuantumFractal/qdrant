use criterion::{black_box, criterion_group, criterion_main, Criterion};
mod prof;

fn fib_vec(n: u64, v: &mut Vec<u64>) {
    match n {
        0 => (),
        1 => v.push(1),
        _ => {
            fib_vec(n - 1, v);
            fib_vec(n - 2, v);
        }
    }
}

pub fn bench_foo(c: &mut Criterion) {
    let mut group = c.benchmark_group("fib");

    group.bench_function("fib20", |b| {
        b.iter(|| {
            let mut v = Vec::new();
            fib_vec(20, &mut v);
            v.iter().sum::<u64>()
        })
    });
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    targets = bench_foo,
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_foo,
}

criterion_main!(benches);
