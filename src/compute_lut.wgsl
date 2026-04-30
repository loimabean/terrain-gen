// SquirrelNoise5 hash
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
    return squirrel5(x + (y * SQ5_PRIME1), seed);
}

// Perlin helpers
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn fade_derivative(t: f32) -> f32 {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

// Gradient LUT lookup
// gradient_lut is 256 pre-computed unit vectors uploaded from the host
// use the low 8 bits of the hash to index into the table, replacing the
// cos/sin pair that the baseline and shared-memory shaders compute per-thread
fn get_gradient(x: i32, y: i32, seed: u32) -> vec2<f32> {
    let hash = squirrel5_2d(x, y, seed);
    return gradient_lut[hash & 255u];
}

// Returns vec3(value, grad_x, grad_y)
fn perlin2d_grad(pos: vec2<f32>, seed: u32) -> vec3<f32> {
    let i = floor(pos);
    let f = pos - i;
    let xi = i32(i.x);
    let yi = i32(i.y);

    let u = fade(f.x);
    let v = fade(f.y);
    let du = fade_derivative(f.x);
    let dv = fade_derivative(f.y);

    let g00 = get_gradient(xi, yi, seed);
    let g10 = get_gradient(xi + 1, yi, seed);
    let g01 = get_gradient(xi, yi + 1, seed);
    let g11 = get_gradient(xi + 1, yi + 1, seed);

    let a = dot(g00, f);
    let b = dot(g10, f - vec2<f32>(1.0, 0.0));
    let c = dot(g01, f - vec2<f32>(0.0, 1.0));
    let d = dot(g11, f - vec2<f32>(1.0, 1.0));

    let val = a + u * (b - a) + v * (c - a) + u * v * (a - b - c + d);
    let grad = g00 + u * (g10 - g00) + v * (g01 - g00) + u * v * (g00 - g10 - g01 + g11)
             + vec2<f32>(du * lerp_v(b - a, d - c, v),
        dv * lerp_v(c - a, d - b, u));

    return vec3<f32>(val, grad);
}

struct ComputeUniforms {
    width: u32,
    height: u32,
    scale: f32,
    seed: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: ComputeUniforms;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gradient_lut: array<vec2<f32>>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= uniforms.width || gid.y >= uniforms.height { return; }

    let pos = vec2<f32>(f32(gid.x), f32(gid.y)) * uniforms.scale;

    var height = 0.0;
    var grad = vec2<f32>(0.0);
    var amplitude = 1.0;
    var frequency = 1.0;

    for (var i = 0u; i < uniforms.octaves; i = i + 1u) {
        let n = perlin2d_grad(pos * frequency, uniforms.seed + i);
        height = height + n.x * amplitude;
        grad = grad + n.yz * amplitude * frequency;

        amplitude = amplitude * uniforms.persistence;
        frequency = frequency * uniforms.lacunarity;
    }

    output_buffer[gid.y * uniforms.width + gid.x] = vec4<f32>(height, grad.x, grad.y, 0.0);
}
