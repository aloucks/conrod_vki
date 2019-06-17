#[macro_use]
extern crate conrod_core;

use vki::{
    winit_surface_descriptor, AdapterOptions, BufferCopyView, BufferDescriptor, BufferUsageFlags, Color,
    DeviceDescriptor, Instance, LoadOp, Origin3D, RenderPassColorAttachmentDescriptor, RenderPassDescriptor, StoreOp,
    SwapchainDescriptor, SwapchainError, TextureCopyView, TextureUsageFlags,
};

use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

mod macros;

widget_ids! {
    #[derive(Debug)]
    struct Ids {
        master,
        left_col,
        middle_col,
        right_col,
        left_text,
        middle_text,
        right_text,
        scrollbar
    }
}

struct Fonts {
    regular: conrod_core::text::font::Id,
    italic: conrod_core::text::font::Id,
    bold: conrod_core::text::font::Id,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = pretty_env_logger::try_init();

    let mut event_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_dimensions((1024, 768).into())
        .with_title("Conrod vki example")
        .with_visibility(false)
        .build(&event_loop)?;

    let dpi_factor = window.get_hidpi_factor();

    let instance = Instance::new()?;
    let surface_descriptor = winit_surface_descriptor!(window);
    let surface = instance.create_surface(&surface_descriptor)?;
    let adapter = instance.get_adapter(AdapterOptions::default())?;
    let device = adapter.create_device(DeviceDescriptor {
        surface_support: Some(&surface),
        ..Default::default()
    })?;
    let swapchain_format = device.get_supported_swapchain_formats(&surface)?[0];
    let swapchain_descriptor = SwapchainDescriptor {
        surface: &surface,
        usage: TextureUsageFlags::OUTPUT_ATTACHMENT,
        format: swapchain_format,
    };
    let mut swapchain = device.create_swapchain(swapchain_descriptor, None)?;

    let mut window_inner_size = window.get_inner_size().unwrap();

    let mut ui = conrod_core::UiBuilder::new([window_inner_size.width as f64, window_inner_size.height as f64]).build();
    let ids = Ids::new(ui.widget_id_generator());
    let image_map = conrod_core::image::Map::new();

    let regular = conrod_core::text::Font::from_bytes(&include_bytes!("fonts/NotoSans/NotoSans-Regular.ttf")[..])?;
    let italic = conrod_core::text::Font::from_bytes(&include_bytes!("fonts/NotoSans/NotoSans-Italic.ttf")[..])?;
    let bold = conrod_core::text::Font::from_bytes(&include_bytes!("fonts/NotoSans/NotoSans-Bold.ttf")[..])?;

    let fonts = Fonts {
        regular: ui.fonts.insert(regular),
        italic: ui.fonts.insert(italic),
        bold: ui.fonts.insert(bold),
    };

    ui.theme.font_id = Some(fonts.regular);

    let mut renderer = conrod_vki::Renderer::new(
        device.clone(),
        swapchain_format,
        window_inner_size.width as u32,
        window_inner_size.height as u32,
    )?;

    let mut edit_text = String::new();

    window.show();

    let mut resize = false;
    let mut running = true;

    while running {
        event_loop.poll_events(|event: winit::Event| {
            if let Some(conrod_event) = convert_event!(event.clone(), &window) {
                ui.handle_event(conrod_event);
            }
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => running = false,
                Event::WindowEvent {
                    event: WindowEvent::Resized(logical_size),
                    ..
                } => {
                    resize = true;
                    window_inner_size = logical_size;
                }
                _ => {}
            }
        });

        if resize {
            swapchain = device.create_swapchain(swapchain_descriptor, Some(&swapchain))?;
            renderer.update_dimensions(window_inner_size.width as u32, window_inner_size.height as u32)?;
            resize = false;
        }

        let viewport = conrod_vki::Viewport {
            x: 0.0,
            y: 0.0,
            width: window_inner_size.width as f32,
            height: window_inner_size.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let mut ui = ui.set_widgets();
        gui(&mut ui, &ids, &fonts, &mut edit_text);

        let mut encoder = device.create_command_encoder()?;

        if let Some(primitives) = ui.draw_if_changed() {
            match renderer.fill(&image_map, viewport, dpi_factor, primitives)? {
                None => {}
                Some(glyph_cache_command) => {
                    let mapped_buffer = device.create_buffer_mapped(BufferDescriptor {
                        usage: BufferUsageFlags::MAP_WRITE | BufferUsageFlags::TRANSFER_SRC,
                        size: glyph_cache_command.glyph_cache_pixel_buffer.len() * std::mem::size_of::<u8>(),
                    })?;
                    mapped_buffer.copy_from_slice(glyph_cache_command.glyph_cache_pixel_buffer)?;
                    let copy_size = glyph_cache_command.glyph_cache_texture.size();
                    encoder.copy_buffer_to_texture(
                        BufferCopyView {
                            offset: 0,
                            image_height: copy_size.height,
                            row_length: copy_size.width,
                            buffer: &mapped_buffer.unmap(),
                        },
                        TextureCopyView {
                            texture: glyph_cache_command.glyph_cache_texture,
                            mip_level: 0,
                            array_layer: 0,
                            origin: Origin3D { x: 0, y: 0, z: 0 },
                        },
                        copy_size,
                    );
                }
            }
        }

        let frame = match swapchain.acquire_next_image() {
            Ok(frame) => frame,
            Err(SwapchainError::OutOfDate) => {
                resize = true;
                continue;
            }
            Err(e) => Err(e)?,
        };

        let mut render_pass = encoder.begin_render_pass(RenderPassDescriptor {
            depth_stencil_attachment: None,
            color_attachments: &[RenderPassColorAttachmentDescriptor {
                resolve_target: None,
                attachment: &frame.view,
                clear_color: Color {
                    r: 0.1,
                    g: 0.1,
                    b: 0.1,
                    a: 1.0,
                },
                store_op: StoreOp::Store,
                load_op: LoadOp::Clear,
            }],
        });

        for cmd in renderer.draw(&image_map, viewport)?.drain(..) {
            let vp = cmd.dynamic_state.viewport;
            let sc = cmd.dynamic_state.scissor;
            render_pass.set_pipeline(&cmd.pipeline);
            render_pass.set_viewport(vp.x, vp.y, vp.width, vp.height, vp.min_depth, vp.max_depth);
            render_pass.set_scissor_rect(sc.x, sc.y, sc.width, sc.height);
            render_pass.set_bind_group(0, &cmd.bind_group, None);
            render_pass.set_vertex_buffers(0, &[cmd.vertex_buffer], &[cmd.vertex_buffer_offset]);
            render_pass.draw(cmd.vertex_count, 1, 0, 0);
        }

        render_pass.end_pass();

        let queue = device.get_queue();

        queue.submit(&[encoder.finish()?])?;

        match queue.present(frame) {
            Ok(_) => {}
            Err(SwapchainError::OutOfDate) => {
                resize = true;
                continue;
            }
            Err(e) => Err(e)?,
        };
    }

    Ok(())
}

fn gui(ui: &mut conrod_core::UiCell, ids: &Ids, fonts: &Fonts, edit_text: &mut String) {
    use conrod_core::{color, widget, Colorable, Positionable, Scalar, Sizeable, Widget};

    // Our `Canvas` tree, upon which we will place our text widgets.
    widget::Canvas::new()
        .flow_right(&[
            (
                ids.left_col,
                widget::Canvas::new().color(color::BLACK).scroll_kids_vertically(),
            ),
            (
                ids.middle_col,
                widget::Canvas::new()
                    .color(color::DARK_CHARCOAL)
                    .scroll_kids_vertically(),
            ),
            (
                ids.right_col,
                widget::Canvas::new().color(color::CHARCOAL).scroll_kids_vertically(),
            ),
        ])
        .set(ids.master, ui);

    const DEMO_TEXT: &'static str =
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
         Mauris aliquet porttitor tellus vel euismod. Integer lobortis volutpat bibendum. Nulla \
         finibus odio nec elit condimentum, rhoncus fermentum purus lacinia. Interdum et malesuada \
         fames ac ante ipsum primis in faucibus. Cras rhoncus nisi nec dolor bibendum pellentesque. \
         Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. \
         Quisque commodo nibh hendrerit nunc sollicitudin sodales. Cras vitae tempus ipsum. Nam \
         magna est, efficitur suscipit dolor eu, consectetur consectetur urna.";

    const PAD: Scalar = 20.0;

    widget::Text::new(DEMO_TEXT)
        .font_id(fonts.regular)
        .color(color::LIGHT_RED)
        .padded_w_of(ids.left_col, PAD)
        .mid_top_with_margin_on(ids.left_col, PAD)
        .left_justify()
        .line_spacing(10.0)
        .set(ids.left_text, ui);

    if edit_text.is_empty() && edit_text.capacity() == 0 {
        edit_text.push_str("Click to edit\n\n");
        edit_text.push_str(DEMO_TEXT);
    }

    for new_text in widget::TextEdit::new(&edit_text)
        .font_id(fonts.italic)
        .color(color::LIGHT_GREEN)
        .mid_left_with_margin_on(ids.middle_col, PAD)
        .line_spacing(10.0)
        .center_justify()
        .restrict_to_height(false)
        .set(ids.middle_text, ui)
    {
        edit_text.clear();
        edit_text.push_str(&new_text);
    }

    widget::Scrollbar::y_axis(ids.middle_col)
        .auto_hide(false)
        .set(ids.scrollbar, ui);

    widget::Text::new(DEMO_TEXT)
        .font_id(fonts.bold)
        .color(color::LIGHT_BLUE)
        .padded_w_of(ids.right_col, PAD)
        .mid_bottom_with_margin_on(ids.right_col, PAD)
        .right_justify()
        .line_spacing(5.0)
        .set(ids.right_text, ui);
}
