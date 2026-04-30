use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cs677_final_project::noise::generate_perlin_grid;

fn bench_noise_resolutions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perlin Noise CPU");

    for size in [128, 256, 512, 1024].iter() {
        group.bench_with_input(format!("{}x{}", size, size), size, |b, &size| {
            b.iter(|| {
                generate_perlin_grid(
                    black_box(size),
                    black_box(size),
                    black_box(0.01),
                    black_box(42),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_noise_resolutions);
criterion_main!(benches);
