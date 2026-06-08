use glam::Vec2;
use std::f32::consts::PI;

// SquirrelNoise5 constants
// http://eiserloh.net/noise/SquirrelNoise5.hpp
// designed by Squirrel Eiserloh
const SQ5_BIT_NOISE1: u32 = 0xd2a80a3f;
const SQ5_BIT_NOISE2: u32 = 0xa884f197;
const SQ5_BIT_NOISE3: u32 = 0x6C736F4B;
const SQ5_BIT_NOISE4: u32 = 0xB79F3ABB;
const SQ5_BIT_NOISE5: u32 = 0x1b56c4f5;
const SQ5_PRIME1: i32 = 198491317;

fn squirrel5(n: i32, seed: u32) -> u32 {
    let mut mangled = n as u32;
    mangled = mangled.wrapping_mul(SQ5_BIT_NOISE1);
    mangled = mangled.wrapping_add(seed);
    mangled ^= mangled >> 9;
    mangled = mangled.wrapping_add(SQ5_BIT_NOISE2);
    mangled ^= mangled >> 11;
    mangled = mangled.wrapping_mul(SQ5_BIT_NOISE3);
    mangled ^= mangled >> 13;
    mangled = mangled.wrapping_add(SQ5_BIT_NOISE4);
    mangled ^= mangled >> 15;
    mangled = mangled.wrapping_mul(SQ5_BIT_NOISE5);
    mangled ^= mangled >> 17;
    mangled
}

fn squirrel5_2d(x: i32, y: i32, seed: u32) -> u32 {
    let n = x.wrapping_add(y.wrapping_mul(SQ5_PRIME1));
    squirrel5(n, seed)
}

fn fade(t: f32) -> f32 {
    // quintic interpolation: 6t^5 - 15t^4 + 10t^3
    // it's what perlin used!
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn get_gradient(x: i32, y: i32, seed: u32) -> Vec2 {
    let hash = squirrel5_2d(x, y, seed);
    let angle = (hash as f32) * 2.0 * PI / (u32::MAX as f32);
    Vec2::new(angle.cos(), angle.sin())
}

// 256 evenly-spaced unit vectors
pub fn build_gradient_lut() -> Vec<Vec2> {
    (0u32..256)
        .map(|i| {
            let angle = (i as f32 / 256.0) * 2.0 * PI;
            Vec2::new(angle.cos(), angle.sin())
        })
        .collect()
}

fn get_gradient_lut(x: i32, y: i32, seed: u32, lut: &[Vec2]) -> Vec2 {
    let hash = squirrel5_2d(x, y, seed);
    lut[(hash & 255) as usize]
}

pub fn perlin2d(x: f32, y: f32, seed: u32) -> f32 {
    let x_i = x.floor() as i32;
    let y_i = y.floor() as i32;
    let x_f = x - x.floor();
    let y_f = y - y.floor();

    let u = fade(x_f);
    let v = fade(y_f);

    let g00 = get_gradient(x_i, y_i, seed);
    let g10 = get_gradient(x_i + 1, y_i, seed);
    let g01 = get_gradient(x_i, y_i + 1, seed);
    let g11 = get_gradient(x_i + 1, y_i + 1, seed);

    let n00 = g00.x * x_f + g00.y * y_f;
    let n10 = g10.x * (x_f - 1.0) + g10.y * y_f;
    let n01 = g01.x * x_f + g01.y * (y_f - 1.0);
    let n11 = g11.x * (x_f - 1.0) + g11.y * (y_f - 1.0);

    lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
}

pub fn generate_perlin_grid(width: u32, height: u32, scale: f32, seed: u32) -> Vec<f32> {
    let mut grid = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let nx = x as f32 * scale;
            let ny = y as f32 * scale;
            grid.push(perlin2d(nx, ny, seed));
        }
    }
    grid
}

pub fn generate_fbm_grid(options: &crate::TerrainOptions) -> Vec<f32> {
    let mut grid = Vec::with_capacity((options.width * options.height) as usize);
    for y in 0..options.height {
        for x in 0..options.width {
            let mut val = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            let nx = x as f32 * options.scale;
            let ny = y as f32 * options.scale;

            for i in 0..options.octaves {
                val += perlin2d(nx * frequency, ny * frequency, options.seed + i) * amplitude;
                amplitude *= options.persistence;
                frequency *= options.lacunarity;
            }
            grid.push(val);
        }
    }
    grid
}

pub fn generate_fbm_grid_lut(options: &crate::TerrainOptions) -> Vec<f32> {
    let lut = build_gradient_lut();
    let mut grid = Vec::with_capacity((options.width * options.height) as usize);
    for y in 0..options.height {
        for x in 0..options.width {
            let mut val = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            let nx = x as f32 * options.scale;
            let ny = y as f32 * options.scale;

            for i in 0..options.octaves {
                val += perlin2d_lut(nx * frequency, ny * frequency, options.seed + i, &lut)
                    * amplitude;
                amplitude *= options.persistence;
                frequency *= options.lacunarity;
            }
            grid.push(val);
        }
    }
    grid
}

fn perlin2d_lut(x: f32, y: f32, seed: u32, lut: &[Vec2]) -> f32 {
    let x_i = x.floor() as i32;
    let y_i = y.floor() as i32;
    let x_f = x - x.floor();
    let y_f = y - y.floor();

    let u = fade(x_f);
    let v = fade(y_f);

    let g00 = get_gradient_lut(x_i, y_i, seed, lut);
    let g10 = get_gradient_lut(x_i + 1, y_i, seed, lut);
    let g01 = get_gradient_lut(x_i, y_i + 1, seed, lut);
    let g11 = get_gradient_lut(x_i + 1, y_i + 1, seed, lut);

    let n00 = g00.x * x_f + g00.y * y_f;
    let n10 = g10.x * (x_f - 1.0) + g10.y * y_f;
    let n01 = g01.x * x_f + g01.y * (y_f - 1.0);
    let n11 = g11.x * (x_f - 1.0) + g11.y * (y_f - 1.0);

    lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_grid_points() {
        assert!((perlin2d(0.0, 0.0, 0).abs() < 1e-6));
        assert!((perlin2d(1.0, 0.0, 42).abs() < 1e-6));
        assert!((perlin2d(0.0, 1.0, 123).abs() < 1e-6));
        assert!((perlin2d(5.0, 5.0, 999).abs() < 1e-6));
    }

    #[test]
    fn test_perlin_continuity() {
        let val1 = perlin2d(0.5, 0.5, 12345);
        let val2 = perlin2d(0.5001, 0.5001, 12345);
        assert!((val1 - val2).abs() < 1e-3);
    }
}
