// SquirrelNoise5 constants
const SQ5_BIT_NOISE1: u32 = 0xd2a80a3fu;
const SQ5_BIT_NOISE2: u32 = 0xa884f197u;
const SQ5_BIT_NOISE3: u32 = 0x6C736F4Bu;
const SQ5_BIT_NOISE4: u32 = 0xB79F3ABBu;
const SQ5_BIT_NOISE5: u32 = 0x1b56c4f5u;
const SQ5_PRIME1: i32 = 198491317;

fn squirrel5(n: i32, seed: u32) -> u32 {
    var mangled = u32(n);
    mangled = mangled * SQ5_BIT_NOISE1;
    mangled = mangled + seed;
    mangled ^= (mangled >> 9u);
    mangled = mangled + SQ5_BIT_NOISE2;
    mangled ^= (mangled >> 11u);
    mangled = mangled * SQ5_BIT_NOISE3;
    mangled ^= (mangled >> 13u);
    mangled = mangled + SQ5_BIT_NOISE4;
    mangled ^= (mangled >> 15u);
    mangled = mangled * SQ5_BIT_NOISE5;
    mangled ^= (mangled >> 17u);
    return mangled;
}

fn squirrel5_2d(x: i32, y: i32, seed: u32) -> u32 {
    let n = x + (y* SQ5_PRIME1);
    return squirrel5(n, seed);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

fn get_gradient(x: i32, y: i32, seed: u32) -> vec2<f32> {
    let hash = squirrel5_2d(x, y, seed);

    let angle = f32(hash) * 2.0 * 3.1415926535 / f32(0xffffffffu);
    return vec2<f32>(cos(angle), sin(angle));
}

fn perlin2d(pos: vec2<f32>, seed: u32) -> f32 {
    let i = floor(pos);
    let f = pos - i;

    let x_i = i32(i.x);
    let y_i = i32(i.y);

    let u = fade(f.x);
    let v = fade(f.y);

    let g00 = get_gradient(x_i, y_i, seed);
    let g10 = get_gradient(x_i + 1, y_i, seed);
    let g01 = get_gradient(x_i, y_i + 1, seed);
    let g11 = get_gradient(x_i + 1, y_i + 1, seed);

    let n00 = dot(g00, f);
    let n10 = dot(g10, f - vec2<f32>(1.0, 0.0));
    let n01 = dot(g01, f - vec2<f32>(0.0, 1.0));
    let n11 = dot(g11, f - vec2<f32>(1.0, 1.0));

    return lerp(
        lerp(n00, n10, u),
        lerp(n01, n11, u),
        v
    );
}

struct ComputeUniforms {
    width: u32,
    height: u32,
    scale: f32,
    seed: u32,
};

@group(0) @binding(0) var<uniform> uniforms: ComputeUniforms;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= uniforms.width || global_id.y >= uniforms.height {
        return;
    }

    let x = f32(global_id.x) * uniforms.scale;
    let y = f32(global_id.y) * uniforms.scale;

    let noise_val = perlin2d(vec2<f32>(x, y), uniforms.seed);

    let index = global_id.y * uniforms.width + global_id.x;
    output_buffer[index] = noise_val;
}
