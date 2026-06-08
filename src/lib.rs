pub mod camera;
pub mod noise;
mod texture;

use camera::{Camera, CameraController, CameraUniform, camera_for_grid};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

fn create_plane_mesh(width: u32, height: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for z in 0..height {
        for x in 0..width {
            vertices.push(Vertex {
                position: [x as f32, 0.0, z as f32],
            });
        }
    }

    for z in 0..height - 1 {
        for x in 0..width - 1 {
            let top_left = z * width + x;
            let top_right = top_left + 1;
            let bottom_left = (z + 1) * width + x;
            let bottom_right = bottom_left + 1;

            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);

            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    (vertices, indices)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainOptions {
    pub width: u32,
    pub height: u32,
    pub scale: f32,
    pub seed: u32,
    pub octaves: u32,
    pub persistence: f32,
    pub lacunarity: f32,
    pub _padding: u32, // struct is 28 bytes without this
}

impl Default for TerrainOptions {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            scale: 0.01,
            seed: 42, // the answer to life, the universe, and everything
            octaves: 6,
            persistence: 0.5,
            lacunarity: 2.0,
            _padding: 0,
        }
    }
}

/// Which terrain generation pipeline is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    /// Pipeline 2 - baseline GPU compute shader (one trig call per gradient).
    ComputeStandard,
    /// Pipeline 3a - GPU compute with shared workgroup memory caching the 1st-octave gradients.
    ComputeOptimized,
    /// Pipeline 3b - GPU compute replacing cos/sin with a 256-entry precomputed LUT.
    ComputeLut,
    /// Pipeline 3c - LUT + all-octave shared-memory cache (single barrier, two loads/thread).
    ComputeCombined,
    /// Pipeline 4 - terrain height computed entirely inside the vertex shader; no compute pass.
    VertexShader,
}

impl PipelineMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ComputeStandard => "Compute - Standard",
            Self::ComputeOptimized => "Compute - Shared Mem",
            Self::ComputeLut => "Compute - Gradient LUT",
            Self::ComputeCombined => "Compute - LUT + Shared Mem",
            Self::VertexShader => "Vertex Shader",
        }
    }

    /// Cycle to the next mode.
    fn next(self) -> Self {
        match self {
            Self::ComputeStandard => Self::ComputeOptimized,
            Self::ComputeOptimized => Self::ComputeLut,
            Self::ComputeLut => Self::ComputeCombined,
            Self::ComputeCombined => Self::VertexShader,
            Self::VertexShader => Self::ComputeStandard,
        }
    }
}

pub struct State {
    // wgpu core
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,

    // geometry
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    // textures
    depth_texture: texture::Texture,

    // camera
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // terrain parameters
    pub terrain_options: TerrainOptions,
    pipeline_mode: PipelineMode,

    // pipeline 2/3: GPU compute -> heightmap -> render
    // (pipeline 1 is CPU noise.rs, used only for correctness verification)
    compute_pipeline: wgpu::ComputePipeline, // standard
    optimized_compute_pipeline: wgpu::ComputePipeline, // shared-mem
    lut_compute_pipeline: wgpu::ComputePipeline, // gradient LUT
    combined_compute_pipeline: wgpu::ComputePipeline, // LUT + all-octave shared mem
    compute_bind_group: wgpu::BindGroup,     // standard + optimized share this
    compute_lut_bind_group: wgpu::BindGroup, // LUT pipeline
    compute_uniform_buffer: wgpu::Buffer,
    compute_output_buffer: wgpu::Buffer,
    compute_read_buffer: wgpu::Buffer,
    gradient_lut_buffer: wgpu::Buffer,

    // stored layouts needed to rebuild bind groups on grid resize
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_lut_bind_group_layout: wgpu::BindGroupLayout,
    terrain_bind_group_layout: wgpu::BindGroupLayout,

    // render pipeline that reads the compute heightmap (pipelines 2 & 3)
    render_pipeline: wgpu::RenderPipeline,
    terrain_bind_group: wgpu::BindGroup,

    // deferred grid resize
    pending_grid_size: Option<(u32, u32)>,

    // pipeline 4: vertex shader inline noise
    vertex_gen_pipeline: wgpu::RenderPipeline,
    vertex_gen_bind_group: wgpu::BindGroup,

    // egui
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    egui_winit_state: egui_winit::State,

    // GPU timestamp queries
    timestamp_query_set: Option<wgpu::QuerySet>,
    timestamp_resolve_buffer: Option<wgpu::Buffer>,
    timestamp_read_buffer: Option<wgpu::Buffer>,
    timestamp_period: f32,
    pending_timestamp: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    gpu_compute_time_ms: Option<f32>,
    gpu_compute_history: std::collections::VecDeque<f32>,

    // performance counters
    frame_timer: web_time::Instant,
    frame_time_ms: f32,
    fps: f32,
    /// Ring buffer of the last 256 per-frame durations (seconds) for a smoothed average.
    fps_history: std::collections::VecDeque<f32>,

    // verification
    verify_requested: bool,
    /// Formatted summary of the last verification run; shown in the egui panel.
    verify_result: String,
    /// WASM only: async verify writes its result here; `update()` polls it each frame.
    #[cfg(target_arch = "wasm32")]
    verify_result_slot: std::sync::Arc<std::sync::Mutex<Option<String>>>,

    // last known cursor position (used by handle_cursor_moved; look deltas come from DeviceEvent)
    cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,

    // winit
    window: Arc<Window>,
}

impl State {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::all(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let has_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: if has_timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits {
                        // Default limit is 2048... my monitor is wider than that :(
                        max_texture_dimension_1d: 4096,
                        max_texture_dimension_2d: 4096,
                        // Used to downgrade to Webgl2, but we need more modern capabilities
                        ..wgpu::Limits::downlevel_defaults()
                    }
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let timestamp_period = queue.get_timestamp_period();
        let (timestamp_query_set, timestamp_resolve_buffer, timestamp_read_buffer) =
            if has_timestamps {
                let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("Timestamp Query Set"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                });
                let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Timestamp Resolve Buffer"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Timestamp Read Buffer"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                (Some(qs), Some(resolve_buf), Some(read_buf))
            } else {
                (None, None, None)
            };

        let terrain_options_tmp = TerrainOptions::default();
        let mut camera = camera_for_grid(
            terrain_options_tmp.width,
            terrain_options_tmp.height,
            config.width as f32 / config.height as f32,
        );
        // aspect is set again below from config, but set it here too for clarity
        camera.aspect = config.width as f32 / config.height as f32;
        let camera_controller = CameraController::new(200.0);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let terrain_options = terrain_options_tmp;
        let (vertices, indices) = create_plane_mesh(terrain_options.width, terrain_options.height);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = indices.len() as u32;

        let buffer_size = (terrain_options.width
            * terrain_options.height
            * std::mem::size_of::<[f32; 4]>() as u32)
            as wgpu::BufferAddress;

        let compute_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let compute_read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Read Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let compute_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Uniform Buffer"),
            contents: bytemuck::cast_slice(&[terrain_options]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // pipeline 2 & 3 (optimized)
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_output_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline (Standard)"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let optimized_compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("compute_optimized.wgsl"));
        let optimized_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline (Shared Mem)"),
                layout: Some(&compute_pipeline_layout),
                module: &optimized_compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // pipeline 3 (LUT)
        // build a 256-entry gradient table: evenly-spaced unit vectors.
        let gradient_lut_data: Vec<[f32; 2]> = (0u32..256)
            .map(|i| {
                let angle = (i as f32 / 256.0) * 2.0 * std::f32::consts::PI;
                [angle.cos(), angle.sin()]
            })
            .collect();

        let gradient_lut_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gradient LUT Buffer"),
            contents: bytemuck::cast_slice(&gradient_lut_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let compute_lut_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute LUT Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_lut_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute LUT Bind Group"),
            layout: &compute_lut_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gradient_lut_buffer.as_entire_binding(),
                },
            ],
        });

        let lut_compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("compute_lut.wgsl"));
        let lut_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LUT Compute Pipeline Layout"),
            bind_group_layouts: &[Some(&compute_lut_bind_group_layout)],
            immediate_size: 0,
        });
        let lut_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline (LUT)"),
                layout: Some(&lut_pipeline_layout),
                module: &lut_compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let combined_compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("compute_combined.wgsl"));
        let combined_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline (LUT + Shared Mem)"),
                layout: Some(&lut_pipeline_layout),
                module: &combined_compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // terrain render resources (pipelines 2 & 3)
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let terrain_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let terrain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &terrain_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("terrain_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),  // group 0
                    Some(&terrain_bind_group_layout), // group 1
                ],
                immediate_size: 0,
            });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // pipeline 4 (vertex shader noise)
        // The vertex gen shader computes the Perlin noise per-vertex, no compute pass
        // It only needs group 2 binding 1 (TerrainOptions), no heightmap
        let vertex_gen_terrain_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vertex_gen_terrain_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 1, // matches @group(2) @binding(1) in the shader
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let vertex_gen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &vertex_gen_terrain_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: compute_uniform_buffer.as_entire_binding(),
            }],
            label: Some("vertex_gen_bind_group"),
        });

        let vertex_gen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Vertex Gen Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),             // group 0
                    Some(&vertex_gen_terrain_bind_group_layout), // group 1
                ],
                immediate_size: 0,
            });

        let vertex_gen_shader =
            device.create_shader_module(wgpu::include_wgsl!("shader_vertex_gen.wgsl"));
        let vertex_gen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vertex Gen Pipeline"),
            layout: Some(&vertex_gen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_gen_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &vertex_gen_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // egui
        let egui_ctx = egui::Context::default();
        let egui_winit_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            vertex_buffer,
            index_buffer,
            num_indices,
            depth_texture,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            terrain_options,
            pipeline_mode: PipelineMode::ComputeStandard,
            compute_pipeline,
            optimized_compute_pipeline,
            lut_compute_pipeline,
            combined_compute_pipeline,
            compute_bind_group,
            compute_lut_bind_group,
            compute_uniform_buffer,
            compute_output_buffer,
            compute_read_buffer,
            gradient_lut_buffer,
            compute_bind_group_layout,
            compute_lut_bind_group_layout,
            terrain_bind_group_layout,
            render_pipeline,
            terrain_bind_group,
            pending_grid_size: None,
            vertex_gen_pipeline,
            vertex_gen_bind_group,
            egui_ctx,
            egui_renderer,
            egui_winit_state,
            timestamp_query_set,
            timestamp_resolve_buffer,
            timestamp_read_buffer,
            timestamp_period,
            pending_timestamp: None,
            gpu_compute_time_ms: None,
            gpu_compute_history: std::collections::VecDeque::with_capacity(256),
            frame_timer: web_time::Instant::now(),
            frame_time_ms: 0.0,
            fps: 0.0,
            fps_history: std::collections::VecDeque::with_capacity(256),
            verify_requested: false,
            verify_result: String::new(),
            #[cfg(target_arch = "wasm32")]
            verify_result_slot: std::sync::Arc::new(std::sync::Mutex::new(None)),
            cursor_pos: None,
            window,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.is_surface_configured = true;
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = width as f32 / height as f32;
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn rebuild_grid(&mut self, new_width: u32, new_height: u32) {
        self.terrain_options.width = new_width;
        self.terrain_options.height = new_height;

        let (vertices, indices) = create_plane_mesh(new_width, new_height);
        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.num_indices = indices.len() as u32;

        let buffer_size = (new_width * new_height * std::mem::size_of::<[f32; 4]>() as u32)
            as wgpu::BufferAddress;
        self.compute_output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.compute_read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Read Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.compute_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.compute_output_buffer.as_entire_binding(),
                },
            ],
        });

        self.compute_lut_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute LUT Bind Group"),
            layout: &self.compute_lut_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.compute_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.compute_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.gradient_lut_buffer.as_entire_binding(),
                },
            ],
        });

        self.terrain_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.terrain_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.compute_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.compute_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("terrain_bind_group"),
        });

        // recenter the camera on the new grid
        let aspect = self.camera.aspect;
        self.camera = camera_for_grid(new_width, new_height, aspect);
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::KeyV, true) => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.verify_result = pollster::block_on(self.verify_gpu())
                        .unwrap_or_else(|e| format!("Error: {e}"));
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let slot = self.verify_result_slot.clone();
                    let state = self as *const State;
                    wasm_bindgen_futures::spawn_local(async move {
                        let msg = unsafe { &*state }
                            .verify_gpu()
                            .await
                            .unwrap_or_else(|e| format!("Error: {e}"));
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(msg);
                        }
                    });
                }
            }
            (KeyCode::KeyO, true) => {
                self.pipeline_mode = self.pipeline_mode.next();
                log::info!("Pipeline: {}", self.pipeline_mode.label());
            }
            (KeyCode::F11 | KeyCode::KeyF, true) => match self.window.fullscreen() {
                None => self
                    .window
                    .set_fullscreen(Some(Fullscreen::Borderless(None))),
                Some(_) => self.window.set_fullscreen(None),
            },
            _ => {
                self.camera_controller.handle_key(code, is_pressed);
            }
        }
    }

    // egui event passthrough

    fn handle_egui_event(&mut self, event: &WindowEvent) -> bool {
        self.egui_winit_state
            .on_window_event(&self.window, event)
            .consumed
    }

    fn set_mouse_grab(&self, grab: bool) {
        use winit::window::CursorGrabMode;
        if grab {
            // locked is ideal, confined is a fallback
            let _ = self
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
            self.window.set_cursor_visible(false);
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
    }

    fn handle_mouse_input(&mut self, button: MouseButton, pressed: bool) {
        if button == MouseButton::Right {
            self.camera_controller.mouse_look_active = pressed;
            self.set_mouse_grab(pressed);
        }
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        self.camera_controller.scroll += match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
        };
    }

    fn handle_cursor_moved(&mut self, pos: winit::dpi::PhysicalPosition<f64>) {
        self.cursor_pos = Some(pos);
    }

    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.camera_controller.mouse_look_active {
            self.camera_controller.rotate_horizontal += dx as f32;
            self.camera_controller.rotate_vertical += dy as f32;
        }
    }

    // compute dispatch

    fn run_compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let (pipeline, bind_group) = match self.pipeline_mode {
            PipelineMode::ComputeStandard => (&self.compute_pipeline, &self.compute_bind_group),
            PipelineMode::ComputeOptimized => {
                (&self.optimized_compute_pipeline, &self.compute_bind_group)
            }
            PipelineMode::ComputeLut => (&self.lut_compute_pipeline, &self.compute_lut_bind_group),
            PipelineMode::ComputeCombined => (
                &self.combined_compute_pipeline,
                &self.compute_lut_bind_group,
            ),
            PipelineMode::VertexShader => return, // terrain generated inline in VS, no compute pass
        };

        let ts_writes =
            self.timestamp_query_set
                .as_ref()
                .map(|qs| wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: ts_writes,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        let wg_x = self.terrain_options.width.div_ceil(16);
        let wg_y = self.terrain_options.height.div_ceil(16);
        compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    async fn execute_compute(&self) -> anyhow::Result<Vec<[f32; 4]>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        self.run_compute(&mut encoder);

        let buffer_size =
            (self.terrain_options.width * self.terrain_options.height * 16) as wgpu::BufferAddress;

        encoder.copy_buffer_to_buffer(
            &self.compute_output_buffer,
            0,
            &self.compute_read_buffer,
            0,
            buffer_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.compute_read_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.compute_read_buffer.unmap();
            Ok(result)
        } else {
            anyhow::bail!("Failed to read compute buffer")
        }
    }

    pub async fn verify_gpu(&self) -> anyhow::Result<String> {
        if self.pipeline_mode == PipelineMode::VertexShader {
            let msg =
                "N/A in Vertex Shader mode\n(no compute output buffer to compare)".to_string();
            log::info!("Verification: {msg}");
            return Ok(msg);
        }

        let total_samples = self.terrain_options.width * self.terrain_options.height;
        log::info!(
            "Verifying {} against CPU baseline [{total_samples} samples]...",
            self.pipeline_mode.label(),
        );

        let t_gpu = web_time::Instant::now();
        let gpu_results = self.execute_compute().await?;
        let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

        let t_cpu = web_time::Instant::now();
        let cpu_results = match self.pipeline_mode {
            PipelineMode::ComputeLut | PipelineMode::ComputeCombined => {
                crate::noise::generate_fbm_grid_lut(&self.terrain_options)
            }
            _ => crate::noise::generate_fbm_grid(&self.terrain_options),
        };
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

        let mut diff_count = 0u32;
        let mut diff_lines = String::new();
        for (i, (g, c)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            if (g[0] - c).abs() > 1e-3 {
                if diff_count < 5 {
                    let x = i as u32 % self.terrain_options.width;
                    let y = i as u32 / self.terrain_options.width;
                    diff_lines
                        .push_str(&format!("diff at ({x},{y}) gpu={:.5} cpu={:.5}\n", g[0], c));
                    log::info!("diff at ({x},{y}): GPU={:.5}  CPU={:.5}", g[0], c);
                }
                diff_count += 1;
            }
        }

        let verdict = if diff_count == 0 {
            format!("PASS: all {total_samples} samples match")
        } else {
            format!("FAIL: {diff_count} / {total_samples} samples differ")
        };

        let summary = format!(
            "Pipeline: {}\nGPU: {gpu_ms:.2} ms\nCPU: {cpu_ms:.1} ms\n{diff_lines}{verdict}",
            self.pipeline_mode.label(),
        );
        log::info!("{summary}");
        Ok(summary)
    }

    fn update(&mut self) {
        // frame timing
        let now = web_time::Instant::now();
        let dt = now.duration_since(self.frame_timer).as_secs_f32();
        self.frame_timer = now;
        self.frame_time_ms = dt * 1000.0;

        // 256-frame average FPS, because the release build runs *really* fast
        if self.fps_history.len() >= 256 {
            self.fps_history.pop_front();
        }
        if dt > 0.0 {
            self.fps_history.push_back(1.0 / dt);
        }
        self.fps = if self.fps_history.is_empty() {
            0.0
        } else {
            self.fps_history.iter().sum::<f32>() / self.fps_history.len() as f32
        };

        // WASM: poll the async verification result slot
        #[cfg(target_arch = "wasm32")]
        if let Ok(mut slot) = self.verify_result_slot.try_lock() {
            if let Some(result) = slot.take() {
                self.verify_result = result;
            }
        }

        // poll for completed GPU timestamp readback
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(rx) = self.pending_timestamp.take() {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    if let Some(read_buf) = &self.timestamp_read_buffer {
                        let data = read_buf.slice(..).get_mapped_range();
                        let ts: &[u64] = bytemuck::cast_slice(&data);
                        let elapsed_ns =
                            ts[1].wrapping_sub(ts[0]) as f64 * self.timestamp_period as f64;
                        drop(data);
                        read_buf.unmap();
                        let new_ms = (elapsed_ns / 1_000_000.0) as f32;
                        if self.gpu_compute_history.len() >= 256 {
                            self.gpu_compute_history.pop_front();
                        }
                        self.gpu_compute_history.push_back(new_ms);
                        self.gpu_compute_time_ms = Some(
                            self.gpu_compute_history.iter().sum::<f32>()
                                / self.gpu_compute_history.len() as f32,
                        );
                    }
                }
                Ok(Err(_)) => {}
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.pending_timestamp = Some(rx); // not ready yet
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }
        }

        // camera
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> anyhow::Result<()> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        // arc clone to mutate fields
        let egui_ctx = self.egui_ctx.clone();
        let raw_input = self.egui_winit_state.take_egui_input(&self.window);

        let ui_output = {
            let options = &mut self.terrain_options;
            let mode = &mut self.pipeline_mode;
            let verify_req = &mut self.verify_requested;
            let grid_size_req = &mut self.pending_grid_size;

            let fps = self.fps;
            let frame_time_ms = self.frame_time_ms;
            let gpu_compute_time_ms = self.gpu_compute_time_ms;
            let verify_result = &self.verify_result;

            egui_ctx.run_ui(raw_input, |ui| {
                egui::Panel::left("terrain_controls")
                    .min_size(250.0)
                    .resizable(true)
                    .show_inside(ui, |ui| {
                        ui.heading("terrain gen!");
                        ui.separator();

                        // perf
                        ui.label(egui::RichText::new("Performance").strong());
                        let fps_color = if fps >= 55.0 {
                            egui::Color32::from_rgb(100, 220, 100)
                        } else if fps >= 30.0 {
                            egui::Color32::from_rgb(220, 180, 50)
                        } else {
                            egui::Color32::from_rgb(220, 80, 80)
                        };
                        ui.horizontal(|ui| {
                            ui.label("FPS:");
                            ui.label(
                                egui::RichText::new(if fps > 0.0 {
                                    format!("{fps:.1}")
                                } else {
                                    "?".to_string()
                                })
                                .color(fps_color)
                                .strong(),
                            );
                            ui.label(
                                egui::RichText::new(if frame_time_ms > 0.0 {
                                    format!("({frame_time_ms:.2} ms)")
                                } else {
                                    "?".to_string()
                                })
                                .weak(),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("GPU compute:");
                            let t_str = if *mode == PipelineMode::VertexShader {
                                "N/A".to_string()
                            } else {
                                gpu_compute_time_ms
                                    .map(|ms| format!("{ms:.3} ms"))
                                    .unwrap_or_else(|| "...".to_string())
                            };
                            ui.label(egui::RichText::new(t_str).monospace());
                        });
                        if ui.button("Copy stats").clicked() {
                            let gpu_t = if *mode == PipelineMode::VertexShader {
                                "N/A".to_string()
                            } else {
                                gpu_compute_time_ms
                                    .map(|ms| format!("{ms:.3} ms"))
                                    .unwrap_or_else(|| "...".to_string())
                            };
                            ui.copy_text(format!(
                                "Pipeline: {}\nGrid: {}x{}\nScale: {:.4}  Octaves: {}  Persistence: {:.2}  Lacunarity: {:.2}  Seed: {}\nFPS: {:.1} ({:.2} ms/frame)\nGPU compute: {}",
                                mode.label(),
                                options.width, options.height,
                                options.scale, options.octaves,
                                options.persistence, options.lacunarity, options.seed,
                                fps, frame_time_ms,
                                gpu_t,
                            ));
                        }

                        ui.separator();

                        // pipeline selector
                        ui.label(egui::RichText::new("Pipeline").strong());
                        for (m, lbl) in [
                            (PipelineMode::ComputeStandard, "Compute - Standard"),
                            (PipelineMode::ComputeOptimized, "Compute - Shared Mem"),
                            (PipelineMode::ComputeLut, "Compute - Gradient LUT"),
                            (PipelineMode::ComputeCombined, "Compute - LUT + Shared Mem"),
                            (PipelineMode::VertexShader, "Vertex Shader"),
                        ] {
                            ui.radio_value(mode, m, lbl);
                        }

                        ui.separator();

                        // noise params
                        ui.label(egui::RichText::new("Noise Parameters").strong());
                        ui.add(
                            egui::Slider::new(&mut options.scale, 0.001..=0.05)
                                .logarithmic(true)
                                .text("Scale"),
                        );
                        ui.add(egui::Slider::new(&mut options.octaves, 1..=16).text("Octaves"));
                        ui.add(
                            egui::Slider::new(&mut options.persistence, 0.1..=0.9)
                                .step_by(0.01)
                                .text("Persistence"),
                        );
                        ui.add(
                            egui::Slider::new(&mut options.lacunarity, 1.0..=4.0)
                                .step_by(0.05)
                                .text("Lacunarity"),
                        );
                        ui.add(egui::Slider::new(&mut options.seed, 0..=9999).text("Seed"));

                        ui.separator();

                        // grid size selector
                        ui.label(egui::RichText::new("Grid Size").strong());
                        const GRID_SIZES: &[u32] = &[64, 128, 256, 512, 1024, 2048];
                        let mut selected = options.width;
                        egui::ComboBox::from_id_salt("grid_size")
                            .selected_text(format!("{}x{}", selected, selected))
                            .show_ui(ui, |ui| {
                                for &s in GRID_SIZES {
                                    ui.selectable_value(&mut selected, s, format!("{}x{}", s, s));
                                }
                            });
                        if selected != options.width {
                            *grid_size_req = Some((selected, selected));
                        }

                        ui.separator();

                        // verify gpu vs cpu results
                        let btn_label = if *mode == PipelineMode::VertexShader {
                            "Verify (N/A, vertex shader pipeline)"
                        } else {
                            "Verify GPU vs CPU"
                        };
                        if ui.button(btn_label).clicked() {
                            *verify_req = true;
                        }

                        if !verify_result.is_empty() {
                            ui.separator();
                            ui.label(egui::RichText::new("Last Verification").strong());
                            egui::ScrollArea::vertical()
                                .id_salt("verify_scroll")
                                .max_height(130.0)
                                .show(ui, |ui| {
                                    ui.label(verify_result.as_str());
                                });
                        }

                        ui.separator();

                        // controls
                        ui.label(egui::RichText::new("Controls").strong());
                        ui.label(egui::RichText::new("WASD/Arrows - move").monospace());
                        ui.label(egui::RichText::new("Space/Shift - up/down").monospace());
                        ui.label(egui::RichText::new("RMB + drag  - look").monospace());
                        ui.label(egui::RichText::new("Scroll      - zoom").monospace());
                        ui.label(egui::RichText::new("O           - cycle pipeline").monospace());
                        ui.label(egui::RichText::new("V           - verify GPU").monospace());
                        ui.label(egui::RichText::new("F11/F       - fullscreen").monospace());
                        ui.label(egui::RichText::new("Esc         - quit").monospace());
                    });
            })
        };

        // handle deferred grid resize before writing terrain_options to GPU
        if let Some((w, h)) = self.pending_grid_size.take() {
            self.rebuild_grid(w, h);
        }

        // cheap enough to write updated params to GPU uniform buffer every frame
        self.queue.write_buffer(
            &self.compute_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.terrain_options]),
        );

        // handle deferred verification request (will block!)
        if self.verify_requested {
            self.verify_requested = false;
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.verify_result =
                    pollster::block_on(self.verify_gpu()).unwrap_or_else(|e| format!("Error: {e}"));
            }
            #[cfg(target_arch = "wasm32")]
            {
                let slot = self.verify_result_slot.clone();
                let state = self as *const State;
                wasm_bindgen_futures::spawn_local(async move {
                    let msg = unsafe { &*state }
                        .verify_gpu()
                        .await
                        .unwrap_or_else(|e| format!("Error: {e}"));
                    if let Ok(mut g) = slot.lock() {
                        *g = Some(msg);
                    }
                });
            }
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                // https://github.com/sotrh/learn-wgpu/issues/668
                drop(surface_texture);
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return Ok(()),
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => anyhow::bail!("Lost device!"),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // compute pass
        self.run_compute(&mut encoder);

        // terrain render pass
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.2,
                            g: 0.3,
                            b: 0.5,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            let (pipeline, terrain_bg) = match self.pipeline_mode {
                PipelineMode::VertexShader => {
                    (&self.vertex_gen_pipeline, &self.vertex_gen_bind_group)
                }
                _ => (&self.render_pipeline, &self.terrain_bind_group),
            };

            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            rpass.set_bind_group(1, terrain_bg, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // egui render pass
        // upload any new/changed font atlas textures.
        for (id, delta) in &ui_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }

        let paint_jobs = egui_ctx.tessellate(ui_output.shapes, ui_output.pixels_per_point);
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ui_output.pixels_per_point,
        };

        // update_buffers may produce extra CommandBuffers from render callbacks
        // submit those before the main encoder to satisfy ordering requirements
        let extra_cbs = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );

        {
            let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // draw on top of terrain
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, // egui doesn't need depth
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            // funky lifetime stuff, but it's fine - will just runtime error if there's an issue
            let mut static_pass = egui_pass.forget_lifetime();
            self.egui_renderer
                .render(&mut static_pass, &paint_jobs, &screen_desc);
        }

        self.egui_winit_state
            .handle_platform_output(&self.window, ui_output.platform_output);

        // free any textures egui no longer needs
        for id in &ui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // resolve timestamp queries into the read buffer (only when compute ran and no readback pending)
        let mut should_map_timestamps = false;
        if self.pipeline_mode != PipelineMode::VertexShader
            && self.pending_timestamp.is_none()
            && let (Some(qs), Some(resolve_buf), Some(read_buf)) = (
                &self.timestamp_query_set,
                &self.timestamp_resolve_buffer,
                &self.timestamp_read_buffer,
            )
        {
            encoder.resolve_query_set(qs, 0..2, resolve_buf, 0);
            encoder.copy_buffer_to_buffer(resolve_buf, 0, read_buf, 0, 16);
            should_map_timestamps = true;
        }

        // callback CBs first, then main encoder
        let mut all_cbs: Vec<wgpu::CommandBuffer> = extra_cbs;
        all_cbs.push(encoder.finish());
        self.queue.submit(all_cbs);

        // kick off async readback of the timestamp buffer
        if should_map_timestamps && let Some(read_buf) = &self.timestamp_read_buffer {
            let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
            read_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.pending_timestamp = Some(rx);
        }

        output.present();

        Ok(())
    }
}

// app boilerplate

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    #[allow(clippy::new_without_default)]
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes().with_title("terrain gen!");

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(State::new(window).await.expect("Unable to create canvas"))
                            .is_ok()
                    )
                })
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };

        let egui_consumed = state.handle_egui_event(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Unable to render: {e}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } if !egui_consumed => {
                state.handle_key(event_loop, code, key_state.is_pressed());
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                state.handle_mouse_input(button, btn_state.is_pressed());
            }
            WindowEvent::MouseWheel { delta, .. } => {
                state.handle_mouse_wheel(delta);
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.handle_cursor_moved(position);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state
            && let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
        {
            state.handle_mouse_motion(dx, dy);
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = App::new();
        event_loop.run_app(&mut app)?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let app = App::new(&event_loop);
        event_loop.spawn_app(app);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
