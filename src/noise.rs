use cgmath::Vector2;
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

fn get_gradient(x: i32, y: i32, seed: u32) -> Vector2<f32> {
    // large prime used to offset the y-axis for 2D hashing
    let hash = squirrel5_2d(x, y, seed);

    // Convert hash to angle in [0, 2PI]
    let angle = (hash as f32) * 2.0 * PI / (u32::MAX as f32);
    Vector2::new(angle.cos(), angle.sin())
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
