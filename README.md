# terrain gen!

This is a procedural terrain generator and renderer built using wgpu and wgsl. Terrain is calculated using Perlin noise layers in a technique known as [fractional Brownian motion](https://en.wikipedia.org/wiki/Fractional_Brownian_motion), and the calculations are done on the GPU with various methods to choose from. Runs both locally and in your browser using WebGPU.

Originally a final project for [CS 677 Parallel Programming for Many-Core Processors](https://web.stevens.edu/catalog/archive/2024-2025/en/catalog/academic-catalog/courses/cs-computer-science/600/cs-677.html) at Stevens Institute of Technology.

## Prerequisites

This program is written in Rust. See [the official Rust website](https://rust-lang.org/) for instructions on how to install it.

Additionally, if you would like to compile to WebAssembly for use in a browser, use the [`wasm-pack` tool](https://github.com/wasm-bindgen/wasm-pack). It can be installed with:
```sh
cargo install wasm-pack
```

## Building & Running

To build and run:
```sh
cargo run
```

To build and run an optimized binary:
```sh
cargo run --release
```

(this might take a while! it has many optimizations enabled)

## Building for the Browser

To build for the browser, run:
```sh
wasm-pack build --target web
```

After that, you can run the browser version using a local server. For example, if you have Python installed, you can run its built-in HTTP server:
```sh
python3 -m http.server
```

## Credits

This project was inspired by Acerola's procedural terrain generator from his video ["Sculpting Terrain with Math."](https://www.youtube.com/watch?v=J1OdPrO7GD0) The source code for that can be found here: [Godot Terrain](https://github.com/GarrettGunnell/Godot-Terrain)

Additionally, the scaffolding/structure of this project was based off of the [Learn Wgpu tutorial.](https://sotrh.github.io/learn-wgpu/)
