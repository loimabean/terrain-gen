use glam::{Mat4, Vec3};
use winit::keyboard::KeyCode;

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,   // radians, 0 = +x, pi/2 = +z
    pub pitch: f32, // radians, positive = up
    pub aspect: f32,
    pub fovy: f32, // degrees
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let target = self.position + self.forward();
        let view = Mat4::look_at_rh(self.position, target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);
        proj * view
    }
}

/// Construct a camera positioned above and behind the center of a grid of the given size.
pub fn camera_for_grid(width: u32, height: u32, aspect: f32) -> Camera {
    let cx = width as f32 / 2.0;
    let cz = height as f32 / 2.0;
    let eye_y = height as f32 * 0.29;
    let eye_z = height as f32 * 0.87;
    let position = Vec3::new(cx, eye_y, eye_z);
    let dir = Vec3::new(cx, 0.0, cz) - position;
    let yaw = dir.z.atan2(dir.x);
    let horiz = (dir.x * dir.x + dir.z * dir.z).sqrt();
    let pitch = dir.y.atan2(horiz);
    Camera {
        position,
        yaw,
        pitch,
        aspect,
        fovy: 45.0,
        znear: 0.1,
        zfar: 10000.0,
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: Mat4,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix();
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CameraController {
    pub speed: f32,
    pub sensitivity: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_left: f32,
    amount_right: f32,
    amount_up: f32,
    amount_down: f32,
    pub rotate_horizontal: f32,
    pub rotate_vertical: f32,
    pub scroll: f32,
    pub mouse_look_active: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            sensitivity: 0.003,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_left: 0.0,
            amount_right: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            mouse_look_active: false,
        }
    }

    /// Returns true if the key was a movement key and was consumed.
    pub fn handle_key(&mut self, key: KeyCode, is_pressed: bool) -> bool {
        let v = if is_pressed { 1.0 } else { 0.0 };
        match key {
            KeyCode::Space => {
                self.amount_up = v;
                true
            }
            KeyCode::ShiftLeft => {
                self.amount_down = v;
                true
            }
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = v;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = v;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = v;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = v;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
        // Horizontal forward (xz plane) for WASD - pitch doesn't affect lateral movement.
        let forward_h = Vec3::new(cos_yaw, 0.0, sin_yaw);
        let right = Vec3::new(-sin_yaw, 0.0, cos_yaw);

        let move_speed = self.speed * dt;
        camera.position += forward_h * (self.amount_forward - self.amount_backward) * move_speed;
        camera.position += right * (self.amount_right - self.amount_left) * move_speed;
        camera.position.y += (self.amount_up - self.amount_down) * move_speed;

        // Scroll moves along the true 3-D forward direction; fixed scale independent of speed.
        camera.position += camera.forward() * self.scroll * 30.0;

        camera.yaw += self.rotate_horizontal * self.sensitivity;
        camera.pitch += -self.rotate_vertical * self.sensitivity;
        let safe = std::f32::consts::FRAC_PI_2 - 0.0001;
        camera.pitch = camera.pitch.clamp(-safe, safe);

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
        self.scroll = 0.0;
    }
}
