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
    return squirrel5(x + (y * SQ5_PRIME1), seed);
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

// Returns vec3(value, grad_x, grad_y)
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

    let val = a + u * (b - a) + v * (c - a) + u * v * (a - b - c + d);

    let grad = g00 + u * (g10 - g00) + v * (g01 - g00) + u * v * (g00 - g10 - g01 + g11) +
               vec2<f32>(du * (lerp_v(b - a, d - c, v)),
        dv * (lerp_v(c - a, d - b, u)));

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

// Shared memory for 1st octave gradients
// 16x16 workgroup, needs up to 18x18 corners for Perlin interpolation
var<workgroup> shared_grads: array<vec2<f32>, 324>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    // 1. Cooperative Load into Shared Memory
    let base_x = i32(floor(f32(group_id.x * 16u) * uniforms.scale));
    let base_y = i32(floor(f32(group_id.y * 16u) * uniforms.scale));

    let local_index = local_id.y * 16u + local_id.x;

    // Load first 256 gradients
    if local_index < 324u {
        let lx = i32(local_index % 18u);
        let ly = i32(local_index / 18u);
        shared_grads[local_index] = get_gradient(base_x + lx, base_y + ly, uniforms.seed);
    }
    // Load remaining 68 gradients
    let second_load = local_index + 256u;
    if second_load < 324u {
        let lx = i32(second_load % 18u);
        let ly = i32(second_load / 18u);
        shared_grads[second_load] = get_gradient(base_x + lx, base_y + ly, uniforms.seed);
    }

    workgroupBarrier();

    if global_id.x >= uniforms.width || global_id.y >= uniforms.height {
        return;
    }

    let world_pos = vec2<f32>(f32(global_id.x), f32(global_id.y)) * uniforms.scale;

    var height = 0.0;
    var grad = vec2<f32>(0.0);

    // First octave using shared memory cache
    {
        let i = floor(world_pos);
        let f = world_pos - i;
        let u = fade(f.x);
        let v = fade(f.y);
        let du = fade_derivative(f.x);
        let dv = fade_derivative(f.y);

        let lx = i32(i.x) - base_x;
        let ly = i32(i.y) - base_y;

        // Ensure we are within the cache
        if lx >= 0 && lx < 17 && ly >= 0 && ly < 17 {
            let g00 = shared_grads[u32(ly * 18 + lx)];
            let g10 = shared_grads[u32(ly * 18 + (lx + 1))];
            let g01 = shared_grads[u32((ly + 1) * 18 + lx)];
            let g11 = shared_grads[u32((ly + 1) * 18 + (lx + 1))];

            let a = dot(g00, f);
            let b = dot(g10, f - vec2<f32>(1.0, 0.0));
            let c = dot(g01, f - vec2<f32>(0.0, 1.0));
            let d = dot(g11, f - vec2<f32>(1.0, 1.0));

            height = a + u * (b - a) + v * (c - a) + u * v * (a - b - c + d);
            grad = g00 + u * (g10 - g00) + v * (g01 - g00) + u * v * (g00 - g10 - g01 + g11) +
                   vec2<f32>(du * (lerp_v(b - a, d - c, v)),
                dv * (lerp_v(c - a, d - b, u)));
        } else {
            // Fallback for safety (should be rare with 18x18 cache and 0.01 scale)
            let n = perlin2d_grad(world_pos, uniforms.seed);
            height = n.x;
            grad = n.yz;
        }
    }

    // Remaining octaves (standard global path)
    var amplitude = uniforms.persistence;
    var frequency = uniforms.lacunarity;

    for (var i = 1u; i < uniforms.octaves; i = i + 1u) {
        let n = perlin2d_grad(world_pos * frequency, uniforms.seed + i);
        height = height + n.x * amplitude;
        grad = grad + n.yz * amplitude * frequency;

        amplitude = amplitude * uniforms.persistence;
        frequency = frequency * uniforms.lacunarity;
    }

    let index = global_id.y * uniforms.width + global_id.x;
    output_buffer[index] = vec4<f32>(height, grad.x, grad.y, 0.0);
}
