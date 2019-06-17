#[macro_use]
extern crate memoffset;

use std::borrow::Cow;

use conrod_core::text::{rt, GlyphCache};
use conrod_core::{color, image, render, Rect, Scalar};

use vki::{
    AddressMode, BindGroup, BindGroupBinding, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutBinding,
    BindGroupLayoutDescriptor, BindingResource, BindingType, BlendDescriptor, Buffer, BufferDescriptor,
    BufferUsageFlags, ColorStateDescriptor, ColorWriteFlags, CompareFunction, CullMode, Device, Extent3D, FilterMode,
    FrontFace, IndexFormat, InputStateDescriptor, InputStepMode, PipelineLayoutDescriptor, PipelineStageDescriptor,
    PrimitiveTopology, RasterizationStateDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderStageFlags, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsageFlags, TextureView, VertexAttributeDescriptor, VertexFormat, VertexInputDescriptor,
};

/// Draw text from the text cache texture `tex` in the fragment shader.
pub const MODE_TEXT: u32 = 0;
/// Draw an image from the texture at `tex` in the fragment shader.
pub const MODE_IMAGE: u32 = 1;
/// Ignore `tex` and draw simple, colored 2D geometry.
pub const MODE_GEOMETRY: u32 = 2;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Scissor {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DynamicState {
    pub scissor: Scissor,
    pub viewport: Viewport,
}

pub struct GlyphCacheCommand<'a> {
    /// The CPU buffer containing the pixel data.
    pub glyph_cache_pixel_buffer: &'a [u8],
    /// The GPU image to which the glyphs are cached.
    pub glyph_cache_texture: &'a Texture,
}

/// A `Command` describing a step in the drawing process.
#[derive(Clone, Debug)]
pub enum Command<'a> {
    /// Draw to the target.
    Draw(Draw<'a>),
    /// Update the scizzor within the pipeline.
    Scizzor(Scissor),
}

/// An iterator yielding `Command`s, produced by the `Renderer::commands` method.
pub struct Commands<'a> {
    commands: std::slice::Iter<'a, PreparedCommand>,
    vertices: &'a [Vertex],
}

impl<'a> Iterator for Commands<'a> {
    type Item = Command<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let Commands {
            ref mut commands,
            ref vertices,
        } = *self;
        commands.next().map(|command| match *command {
            PreparedCommand::Scizzor(scizzor) => Command::Scizzor(scizzor),
            PreparedCommand::Plain(ref range) => Command::Draw(Draw::Plain(&vertices[range.clone()])),
            PreparedCommand::Image(id, ref range) => Command::Draw(Draw::Image(id, &vertices[range.clone()])),
        })
    }
}

/// A `Command` for drawing to the target.
///
/// Each variant describes how to draw the contents of the vertex buffer.
#[derive(Clone, Debug)]
pub enum Draw<'a> {
    /// A range of vertices representing triangles textured with the image in the
    /// image_map at the given `widget::Id`.
    Image(image::Id, &'a [Vertex]),
    /// A range of vertices representing plain triangles.
    Plain(&'a [Vertex]),
}

enum PreparedCommand {
    Image(image::Id, std::ops::Range<usize>),
    Plain(std::ops::Range<usize>),
    Scizzor(Scissor),
}

/// The `Vertex` type passed to the vertex shader.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// The position of the vertex within vector space.
    ///
    /// [-1.0, 1.0] is the leftmost, bottom position of the display.
    /// [1.0, -1.0] is the rightmost, top position of the display.
    pub pos: [f32; 2],
    /// The coordinates of the texture used by this `Vertex`.
    ///
    /// [0.0, 0.0] is the leftmost, top position of the texture.
    /// [1.0, 1.0] is the rightmost, bottom position of the texture.
    pub uv: [f32; 2],
    /// A color associated with the `Vertex`.
    ///
    /// The way that the color is used depends on the `mode`.
    pub color: [f32; 4],
    /// The mode with which the `Vertex` will be drawn within the fragment shader.
    ///
    /// `0` for rendering text.
    /// `1` for rendering an image.
    /// `2` for rendering non-textured 2D geometry.
    ///
    /// If any other value is given, the fragment shader will not output any color.
    pub mode: u32,
}

/// A type used for translating `render::Primitives` into `Command`s that indicate how to draw the
/// conrod GUI using `vulkano`.
pub struct Renderer {
    device: Device,
    pipeline: RenderPipeline,
    glyph_cache: GlyphCache<'static>,
    glyph_cache_texture: Texture,
    glyph_cache_bind_group: BindGroup,
    glyph_cache_pixel_buffer: Vec<u8>,
    bind_group_layout: BindGroupLayout,
    image_bind_groups: image::HashMap<BindGroup>,
    sampler: Sampler,
    commands: Vec<PreparedCommand>,
    vertices: Vec<Vertex>,
    target_width: u32,
    target_height: u32,
}

pub struct DrawCommand {
    pub pipeline: RenderPipeline,
    pub dynamic_state: DynamicState,
    pub bind_group: BindGroup,
    pub vertex_buffer: Buffer,
    pub vertex_buffer_offset: u64,
    pub vertex_count: u32,
}

impl Renderer {
    pub fn new(
        device: Device,
        target_format: TextureFormat,
        target_width: u32,
        target_height: u32,
    ) -> Result<Renderer, vki::Error> {
        let sampler = device.create_sampler(SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare_function: CompareFunction::Never,
        })?;

        let vs = device.create_shader_module(ShaderModuleDescriptor {
            code: Cow::Borrowed(include_bytes!("shaders/conrod.vert.spv")),
        })?;

        let fs = device.create_shader_module(ShaderModuleDescriptor {
            code: Cow::Borrowed(include_bytes!("shaders/conrod.frag.spv")),
        })?;

        let bind_group_layout = device.create_bind_group_layout(BindGroupLayoutDescriptor {
            bindings: &[
                BindGroupLayoutBinding {
                    binding: 0,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::Sampler,
                },
                BindGroupLayoutBinding {
                    binding: 1,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::SampledTexture,
                },
            ],
        })?;

        let pipeline_layout = device.create_pipeline_layout(PipelineLayoutDescriptor {
            push_constant_ranges: vec![],
            bind_group_layouts: vec![bind_group_layout.clone()],
        })?;

        let pipeline = device.create_render_pipeline(RenderPipelineDescriptor {
            layout: pipeline_layout,
            primitive_topology: PrimitiveTopology::TriangleList,
            fragment_stage: PipelineStageDescriptor {
                module: fs,
                entry_point: Cow::Borrowed("main"),
            },
            vertex_stage: PipelineStageDescriptor {
                module: vs,
                entry_point: Cow::Borrowed("main"),
            },
            sample_count: 1,
            depth_stencil_state: None,
            color_states: vec![ColorStateDescriptor {
                write_mask: ColorWriteFlags::ALL,
                color_blend: BlendDescriptor::BLEND,
                alpha_blend: BlendDescriptor::BLEND,
                format: target_format,
            }],
            rasterization_state: RasterizationStateDescriptor {
                front_face: FrontFace::Cw,
                cull_mode: CullMode::None,
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_scale: 0.0,
            },
            input_state: InputStateDescriptor {
                index_format: IndexFormat::U16, // None
                inputs: vec![VertexInputDescriptor {
                    input_slot: 0,
                    step_mode: InputStepMode::Vertex,
                    stride: std::mem::size_of::<Vertex>(),
                }],
                attributes: vec![
                    VertexAttributeDescriptor {
                        input_slot: 0,
                        shader_location: 0,
                        format: VertexFormat::Float2,
                        offset: offset_of!(Vertex, pos) as _,
                    },
                    VertexAttributeDescriptor {
                        input_slot: 0,
                        shader_location: 1,
                        format: VertexFormat::Float2,
                        offset: offset_of!(Vertex, uv) as _,
                    },
                    VertexAttributeDescriptor {
                        input_slot: 0,
                        shader_location: 2,
                        format: VertexFormat::Float4,
                        offset: offset_of!(Vertex, color) as _,
                    },
                    VertexAttributeDescriptor {
                        input_slot: 0,
                        shader_location: 3,
                        format: VertexFormat::UInt,
                        offset: offset_of!(Vertex, mode) as _,
                    },
                ],
            },
        })?;

        let (glyph_cache, glyph_cache_pixel_buffer, glyph_cache_bind_group, glyph_cache_texture) =
            build_glyph_cache(&device, &bind_group_layout, &sampler, target_width, target_height)?;

        let image_bind_groups = image::HashMap::default();
        let commands = Vec::default();
        let vertices = Vec::default();

        Ok(Renderer {
            device,
            pipeline,
            glyph_cache,
            glyph_cache_texture,
            glyph_cache_bind_group,
            glyph_cache_pixel_buffer,
            bind_group_layout,
            image_bind_groups,
            sampler,
            commands,
            vertices,
            target_width,
            target_height,
        })
    }

    pub fn update_dimensions(&mut self, target_width: u32, target_height: u32) -> Result<(), vki::Error> {
        let (glyph_cache, glyph_cache_pixel_buffer, glyph_cache_bind_group, glyph_cache_texture) = build_glyph_cache(
            &self.device,
            &self.bind_group_layout,
            &self.sampler,
            target_width,
            target_height,
        )?;
        self.glyph_cache = glyph_cache;
        self.glyph_cache_pixel_buffer = glyph_cache_pixel_buffer;
        self.glyph_cache_bind_group = glyph_cache_bind_group;
        self.glyph_cache_texture = glyph_cache_texture;
        self.target_width = target_width;
        self.target_height = target_height;
        Ok(())
    }

    /// Produce an `Iterator` yielding `Command`s.
    pub fn commands(&self) -> Commands {
        Commands {
            commands: self.commands.iter(),
            vertices: &self.vertices,
        }
    }

    pub fn draw(
        &mut self,
        image_map: &image::Map<TextureView>,
        viewport: Viewport,
    ) -> Result<Vec<DrawCommand>, vki::Error> {
        let mut draw_commands = Vec::with_capacity(self.commands.len());

        let mut current_scissor = Scissor {
            x: viewport.x.round() as _,
            y: viewport.y.round() as _,
            width: viewport.width.round() as _,
            height: viewport.height.round() as _,
        };

        let dynamic_state = |scissor| DynamicState {
            viewport: viewport.clone(),
            scissor,
        };

        let commands = Commands {
            commands: self.commands.iter(),
            vertices: &self.vertices,
        };

        for command in commands {
            match command {
                Command::Scizzor(scizzor) => {
                    current_scissor = scizzor;
                }
                Command::Draw(draw) => match draw {
                    Draw::Plain(verts) => {
                        if verts.is_empty() {
                            continue;
                        }

                        let mapped_buffer = self.device.create_buffer_mapped(BufferDescriptor {
                            usage: BufferUsageFlags::MAP_WRITE | BufferUsageFlags::VERTEX,
                            size: verts.len() * std::mem::size_of::<Vertex>(),
                        })?;

                        mapped_buffer.copy_from_slice(verts)?;

                        draw_commands.push(DrawCommand {
                            pipeline: self.pipeline.clone(),
                            dynamic_state: dynamic_state(current_scissor.clone()),
                            vertex_buffer: mapped_buffer.unmap(),
                            vertex_buffer_offset: 0, // TODO
                            bind_group: self.glyph_cache_bind_group.clone(),
                            vertex_count: verts.len() as _,
                        });
                    }
                    Draw::Image(image_id, verts) => {
                        if verts.is_empty() {
                            continue;
                        }

                        if let Some(image) = image_map.get(&image_id) {
                            let mapped_buffer = self.device.create_buffer_mapped(BufferDescriptor {
                                usage: BufferUsageFlags::MAP_WRITE | BufferUsageFlags::VERTEX,
                                size: verts.len() * std::mem::size_of::<Vertex>(),
                            })?;

                            mapped_buffer.copy_from_slice(verts)?;

                            let bind_group = if let Some(bind_group) = self.image_bind_groups.get(&image_id).cloned() {
                                bind_group
                            } else {
                                let bind_group = self.device.create_bind_group(BindGroupDescriptor {
                                    layout: self.bind_group_layout.clone(),
                                    bindings: vec![
                                        BindGroupBinding {
                                            binding: 0,
                                            resource: BindingResource::Sampler(self.sampler.clone()),
                                        },
                                        BindGroupBinding {
                                            binding: 1,
                                            resource: BindingResource::TextureView(image.clone()),
                                        },
                                    ],
                                })?;
                                self.image_bind_groups.insert(image_id, bind_group.clone());
                                bind_group
                            };

                            draw_commands.push(DrawCommand {
                                pipeline: self.pipeline.clone(),
                                dynamic_state: dynamic_state(current_scissor.clone()),
                                vertex_buffer: mapped_buffer.unmap(),
                                vertex_buffer_offset: 0, // TODO
                                bind_group,
                                vertex_count: verts.len() as _,
                            });
                        }
                    }
                },
            }
        }

        Ok(draw_commands)
    }

    pub fn fill<P>(
        &mut self,
        image_map: &image::Map<TextureView>,
        viewport: Viewport,
        dpi_factor: f64,
        mut primitives: P,
    ) -> Result<Option<GlyphCacheCommand>, rt::gpu_cache::CacheWriteErr>
    where
        P: render::PrimitiveWalker,
    {
        let Renderer {
            ref mut commands,
            ref mut vertices,
            ref mut glyph_cache,
            ref mut glyph_cache_pixel_buffer,
            ref glyph_cache_texture,
            target_width,
            target_height,
            ..
        } = *self;

        enum State {
            Image { image_id: image::Id, start: usize },
            Plain { start: usize },
        }

        let mut current_state = State::Plain { start: 0 };

        commands.clear();
        vertices.clear();

        // Switches to the `Plain` state and completes the previous `Command` if not already in the
        // `Plain` state.
        macro_rules! switch_to_plain_state {
            () => {
                match current_state {
                    State::Plain { .. } => (),
                    State::Image { image_id, start } => {
                        commands.push(PreparedCommand::Image(image_id, start..vertices.len()));
                        current_state = State::Plain {
                            start: vertices.len(),
                        };
                    }
                }
            };
        }

        let mut update_glyph_cache_texture = false;

        let glyph_cache_w = glyph_cache_texture.size().width;

        let half_win_w = (target_width as f64) / 2.0;
        let half_win_h = (target_height as f64) / 2.0;

        // Functions for converting for conrod scalar coords to Vulkan vertex coords (-1.0 to 1.0)
        // with inverted y
        let vx = |x: Scalar| (x * dpi_factor / half_win_w) as f32;
        let vy = |y: Scalar| (-y * dpi_factor / half_win_h) as f32;

        let mut current_scizzor = Scissor {
            x: 0 as _,
            y: 0 as _,
            width: target_width as _,
            height: target_height as _,
        };

        let rect_to_scissor = |rect: Rect| {
            let (w, h) = rect.w_h();
            let left = (rect.left() * dpi_factor + half_win_w) as u32;
            let bottom = (rect.bottom() * dpi_factor + half_win_h) as u32;
            let width = (w * dpi_factor) as u32;
            let height = (h * dpi_factor) as u32;
            Scissor {
                x: std::cmp::max(left, 0),
                y: std::cmp::max(bottom, 0),
                width: std::cmp::min(width, viewport.width as _),
                height: std::cmp::min(height, viewport.height as _),
            }
        };

        while let Some(primitive) = primitives.next_primitive() {
            let render::Primitive {
                kind, scizzor, rect, ..
            } = primitive;

            let new_scizzor = rect_to_scissor(scizzor);

            if new_scizzor != current_scizzor {
                // Finish the current command.

                match current_state {
                    State::Plain { start } => commands.push(PreparedCommand::Plain(start..vertices.len())),
                    State::Image { image_id, start } => {
                        commands.push(PreparedCommand::Image(image_id, start..vertices.len()))
                    }
                }

                // Update the scizzor and produce a command.
                current_scizzor = new_scizzor;
                commands.push(PreparedCommand::Scizzor(new_scizzor));

                // Set the state back to plain drawing.
                current_state = State::Plain { start: vertices.len() };
            }

            match kind {
                render::PrimitiveKind::Rectangle { color } => {
                    switch_to_plain_state!();

                    let color = gamma_srgb_to_linear(color.to_fsa());
                    let (l, r, b, t) = rect.l_r_b_t();

                    let v = |x, y| {
                        // Convert from conrod Scalar range to GL range -1.0 to 1.0.
                        Vertex {
                            pos: [vx(x), vy(y)],
                            uv: [0.0, 0.0],
                            color,
                            mode: MODE_GEOMETRY,
                        }
                    };

                    let mut push_v = |x, y| vertices.push(v(x, y));

                    // Bottom left triangle.
                    push_v(l, t);
                    push_v(r, b);
                    push_v(l, b);

                    // Top right triangle.
                    push_v(l, t);
                    push_v(r, b);
                    push_v(r, t);
                }

                render::PrimitiveKind::TrianglesSingleColor { color, triangles } => {
                    if triangles.is_empty() {
                        continue;
                    }

                    switch_to_plain_state!();

                    let color = gamma_srgb_to_linear(color.into());

                    let v = |p: [Scalar; 2]| Vertex {
                        pos: [vx(p[0]), vy(p[1])],
                        uv: [0.0, 0.0],
                        color,
                        mode: MODE_GEOMETRY,
                    };

                    for triangle in triangles {
                        vertices.push(v(triangle[0]));
                        vertices.push(v(triangle[1]));
                        vertices.push(v(triangle[2]));
                    }
                }

                render::PrimitiveKind::TrianglesMultiColor { triangles } => {
                    if triangles.is_empty() {
                        continue;
                    }

                    switch_to_plain_state!();

                    let v = |(p, c): ([Scalar; 2], color::Rgba)| Vertex {
                        pos: [vx(p[0]), vy(p[1])],
                        uv: [0.0, 0.0],
                        color: gamma_srgb_to_linear(c.into()),
                        mode: MODE_GEOMETRY,
                    };

                    for triangle in triangles {
                        vertices.push(v(triangle[0]));
                        vertices.push(v(triangle[1]));
                        vertices.push(v(triangle[2]));
                    }
                }

                render::PrimitiveKind::Text { color, text, font_id } => {
                    switch_to_plain_state!();

                    let positioned_glyphs = text.positioned_glyphs(dpi_factor as f32);

                    // Queue the glyphs to be cached
                    for glyph in positioned_glyphs {
                        glyph_cache.queue_glyph(font_id.index(), glyph.clone());
                    }

                    glyph_cache.cache_queued(|rect, data| {
                        let width = (rect.max.x - rect.min.x) as usize;
                        let height = (rect.max.y - rect.min.y) as usize;
                        let mut dst_ix = rect.min.y as usize * glyph_cache_w as usize + rect.min.x as usize;
                        let mut src_ix = 0;
                        for _ in 0..height {
                            let dst_range = dst_ix..dst_ix + width;
                            let src_range = src_ix..src_ix + width;
                            let dst_slice = &mut glyph_cache_pixel_buffer[dst_range];
                            let src_slice = &data[src_range];
                            dst_slice.copy_from_slice(src_slice);
                            dst_ix += glyph_cache_w as usize;
                            src_ix += width;
                        }
                        update_glyph_cache_texture = true;
                    })?;

                    let color = gamma_srgb_to_linear(color.to_fsa());
                    let cache_id = font_id.index();
                    let origin = rt::point(0.0, 0.0);

                    // A closure to convert RustType rects to GL rects
                    let to_vk_rect = |screen_rect: rt::Rect<i32>| rt::Rect {
                        min: origin
                            + (rt::vector(
                                screen_rect.min.x as f32 / viewport.width - 0.5,
                                screen_rect.min.y as f32 / viewport.height - 0.5,
                            )) * 2.0,
                        max: origin
                            + (rt::vector(
                                screen_rect.max.x as f32 / viewport.width - 0.5,
                                screen_rect.max.y as f32 / viewport.height - 0.5,
                            )) * 2.0,
                    };

                    for g in positioned_glyphs {
                        if let Ok(Some((uv_rect, screen_rect))) = glyph_cache.rect_for(cache_id, g) {
                            let vk_rect = to_vk_rect(screen_rect);
                            let v = |p, t| Vertex {
                                pos: p,
                                uv: t,
                                color,
                                mode: MODE_TEXT,
                            };
                            let mut push_v = |p, t| vertices.push(v(p, t));
                            push_v([vk_rect.min.x, vk_rect.max.y], [uv_rect.min.x, uv_rect.max.y]);
                            push_v([vk_rect.min.x, vk_rect.min.y], [uv_rect.min.x, uv_rect.min.y]);
                            push_v([vk_rect.max.x, vk_rect.min.y], [uv_rect.max.x, uv_rect.min.y]);
                            push_v([vk_rect.max.x, vk_rect.min.y], [uv_rect.max.x, uv_rect.min.y]);
                            push_v([vk_rect.max.x, vk_rect.max.y], [uv_rect.max.x, uv_rect.max.y]);
                            push_v([vk_rect.min.x, vk_rect.max.y], [uv_rect.min.x, uv_rect.max.y]);
                        }
                    }
                }

                render::PrimitiveKind::Image {
                    image_id,
                    color,
                    source_rect,
                } => {
                    let image_ref = match image_map.get(&image_id) {
                        None => continue,
                        Some(img) => img,
                    };

                    // Switch to the `Image` state for this image if we're not in it already.
                    let new_image_id = image_id;
                    match current_state {
                        // If we're already in the drawing mode for this image, we're done.
                        State::Image { image_id, .. } if image_id == new_image_id => (),

                        // If we were in the `Plain` drawing state, switch to Image drawing state.
                        State::Plain { start } => {
                            commands.push(PreparedCommand::Plain(start..vertices.len()));
                            current_state = State::Image {
                                image_id: new_image_id,
                                start: vertices.len(),
                            };
                        }

                        // If we were drawing a different image, switch state to draw *this* image.
                        State::Image { image_id, start } => {
                            commands.push(PreparedCommand::Image(image_id, start..vertices.len()));
                            current_state = State::Image {
                                image_id: new_image_id,
                                start: vertices.len(),
                            };
                        }
                    }

                    let image_extent = image_ref.texture().size();

                    let color = color.unwrap_or(color::WHITE).to_fsa();
                    let (image_w, image_h) = (image_extent.width, image_extent.height);
                    let (image_w, image_h) = (image_w as Scalar, image_h as Scalar);

                    // Get the sides of the source rectangle as uv coordinates.
                    //
                    // Texture coordinates range:
                    // - left to right: 0.0 to 1.0
                    // - bottom to top: 0.0 to 0.1
                    // Note bottom and top are flipped in comparison to glium so that we don't need
                    //  to flip images when loading
                    let (uv_l, uv_r, uv_t, uv_b) = match source_rect {
                        Some(src_rect) => {
                            let (l, r, b, t) = src_rect.l_r_b_t();
                            (
                                (l / image_w) as f32,
                                (r / image_w) as f32,
                                (t / image_h) as f32,
                                (b / image_h) as f32,
                            )
                        }
                        None => (0.0, 1.0, 0.0, 1.0),
                    };

                    let v = |x, y, t| {
                        // Convert from conrod Scalar range to GL range -1.0 to 1.0.
                        let x = (x * dpi_factor / half_win_w) as f32;
                        let y = -((y * dpi_factor / half_win_h) as f32);
                        Vertex {
                            pos: [x, y],
                            uv: t,
                            color,
                            mode: MODE_IMAGE,
                        }
                    };

                    let mut push_v = |x, y, t| vertices.push(v(x, y, t));

                    // Swap bottom and top to suit reversed vulkan coords.
                    let (l, r, b, t) = rect.l_r_b_t();

                    // Bottom left triangle.
                    push_v(l, t, [uv_l, uv_t]);
                    push_v(r, b, [uv_r, uv_b]);
                    push_v(l, b, [uv_l, uv_b]);

                    // Top right triangle.
                    push_v(l, t, [uv_l, uv_t]);
                    push_v(r, b, [uv_r, uv_b]);
                    push_v(r, t, [uv_r, uv_t]);
                }

                // We have no special case widgets to handle.
                render::PrimitiveKind::Other(_) => (),
            }
        }

        // Enter the final command.
        match current_state {
            State::Plain { start } => commands.push(PreparedCommand::Plain(start..vertices.len())),
            State::Image { image_id, start } => commands.push(PreparedCommand::Image(image_id, start..vertices.len())),
        }

        let glyph_cache_cmd = if update_glyph_cache_texture {
            Some(GlyphCacheCommand {
                glyph_cache_texture,
                glyph_cache_pixel_buffer,
            })
        } else {
            None
        };

        Ok(glyph_cache_cmd)
    }
}

fn build_glyph_cache(
    device: &Device,
    bind_group_layout: &BindGroupLayout,
    sampler: &Sampler,
    width: u32,
    height: u32,
) -> Result<(GlyphCache<'static>, Vec<u8>, BindGroup, Texture), vki::Error> {
    let cache = GlyphCache::builder()
        .dimensions(width, height)
        .scale_tolerance(0.1)
        .position_tolerance(0.1)
        .build();

    let texture = device.create_texture(TextureDescriptor {
        format: TextureFormat::R8Unorm,
        dimension: TextureDimension::D2,
        usage: TextureUsageFlags::TRANSFER_DST | TextureUsageFlags::SAMPLED,
        mip_level_count: 1,
        array_layer_count: 1,
        sample_count: 1,
        size: Extent3D {
            width,
            height,
            depth: 1,
        },
    })?;

    let texture_view = texture.create_default_view()?;

    let bind_group = device.create_bind_group(BindGroupDescriptor {
        layout: bind_group_layout.clone(),
        bindings: vec![
            BindGroupBinding {
                binding: 0,
                resource: BindingResource::Sampler(sampler.clone()),
            },
            BindGroupBinding {
                binding: 1,
                resource: BindingResource::TextureView(texture_view),
            },
        ],
    })?;

    let pixel_buffer = vec![0; width as usize * height as usize];

    Ok((cache, pixel_buffer, bind_group, texture))
}

fn gamma_srgb_to_linear(c: [f32; 4]) -> [f32; 4] {
    fn component(f: f32) -> f32 {
        // Taken from https://github.com/PistonDevelopers/graphics/src/color.rs#L42
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    }
    [component(c[0]), component(c[1]), component(c[2]), c[3]]
}
