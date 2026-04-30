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

// Terrain heightmap and gradients from the compute shader
@group(1) @binding(0)
var<storage, read> heightmap: array<vec4<f32>>;
@group(1) @binding(1)
var<uniform> options: TerrainOptions;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    let x = u32(model.position.x);
    let z = u32(model.position.z);
    let index = z * options.width + x;

    let height_data = heightmap[index];
    let height = height_data.x;
    let grad = height_data.yz;

    // Construct normal from analytical gradients
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

    // Slope-based coloring
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
