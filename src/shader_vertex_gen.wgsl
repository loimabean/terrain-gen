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
    let n = x + (y * SQ5_PRIME1);
    return squirrel5(n, seed);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn fade_derivative(t: f32) -> f32 {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

fn get_gradient(x: i32, y: i32, seed: u32) -> vec2<f32> {
    let hash = squirrel5_2d(x, y, seed);

    let angle = f32(hash) * 2.0 * 3.1415926535 / f32(0xffffffffu);
    return vec2<f32>(cos(angle), sin(angle));
}

fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

fn perlin2d_grad(pos: vec2<f32>, seed: u32) -> vec3<f32> {
    let i = floor(pos);
    let f = pos - i;
    let x_i = i32(i.x);
    let y_i = i32(i.y);
    let u = fade(f.x);
    let v = fade(f.y);
    let du = fade_derivative(f.x);
    let dv = fade_derivative(f.y);
    let g00 = get_gradient(x_i, y_i, seed);
    let g10 = get_gradient(x_i + 1, y_i, seed);
    let g01 = get_gradient(x_i, y_i + 1, seed);
    let g11 = get_gradient(x_i + 1, y_i + 1, seed);
    let a = dot(g00, f);
    let b = dot(g10, f - vec2<f32>(1.0, 0.0));
    let c = dot(g01, f - vec2<f32>(0.0, 1.0));
    let d = dot(g11, f - vec2<f32>(1.0, 1.0));
    let val = a + u*(b-a) + v*(c-a) + u*v*(a-b-c+d);
    let grad = g00 + u*(g10-g00) + v*(g01-g00) + u*v*(g00-g10-g01+g11) +
               vec2<f32>(du * (lerp_v(b-a, d-c, v)),
                         dv * (lerp_v(c-a, d-b, u)));
    return vec3<f32>(val, grad);
}

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct TerrainOptions {
    width: u32,
    height: u32,
    scale: f32,
    seed: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    _padding: u32,
};
@group(1) @binding(1)
var<uniform> options: TerrainOptions;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    let pos = vec2<f32>(model.position.x, model.position.z) * options.scale;

    var height = 0.0;
    var grad = vec2<f32>(0.0);
    var amplitude = 1.0;
    var frequency = 1.0;

    for (var i = 0u; i < options.octaves; i = i + 1u) {
        let n = perlin2d_grad(pos * frequency, options.seed + i);
        height = height + n.x * amplitude;
        grad = grad + n.yz * amplitude * frequency;

        amplitude = amplitude * options.persistence;
        frequency = frequency * options.lacunarity;
    }

    let world_normal = normalize(vec3<f32>(-grad.x, 1.0, -grad.y));

    var out: VertexOutput;
    let world_pos = vec3<f32>(model.position.x, height * 20.0, model.position.z);
    out.world_normal = world_normal;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let ambient_strength = 0.2;

    let slope = 1.0 - in.world_normal.y;
    var color: vec3<f32>;
    if (slope < 0.1) {
        color = vec3<f32>(0.2, 0.8, 0.2); // Grass
    } else if (slope < 0.4) {
        color = vec3<f32>(0.5, 0.4, 0.3); // Dirt/Rock
    } else {
        color = vec3<f32>(0.9, 0.9, 1.0); // Snow
    }

    let lighting = diffuse_strength + ambient_strength;
    return vec4<f32>(color * lighting, 1.0);
}
