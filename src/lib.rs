mod pipeline_cache;
mod sync_world;

use crate::sync_world::{SyncToRenderWorld, entity_sync_system};
use bevy::ecs::schedule::{ScheduleBuildSettings, ScheduleLabel};
use bevy::ecs::system::SystemState;
use bevy::log::tracing;
use bevy::prelude::*;
use bevy::render::render_resource::{PipelineCache, ShaderLoader};
use bevy::render::renderer::render_system;
use bevy::render::sync_world::TemporaryRenderEntity;
use bevy::render::{Extract, MainWorld, Render, RenderApp, RenderSet, render_graph};
use bevy::window::{PrimaryWindow, RawHandleWrapperHolder};
use std::ops::{Deref, DerefMut};
use vulkano_util::context::VulkanoConfig;

pub struct VulkanRenderPlugin {}

/// Schedule which extract data from the main world and inserts it into the render world.
///
/// This step should be kept as short as possible to increase the "pipelining potential" for
/// running the next frame while rendering the current frame.
///
/// This schedule is run on the main world, but its buffers are not applied
/// until it is returned to the render world.
#[derive(ScheduleLabel, PartialEq, Eq, Debug, Clone, Hash, Default)]
pub struct ExtractSchedule;

pub mod graph {
    use crate::render_graph::RenderLabel;

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub struct CameraDriverLabel;
}

/// A "scratch" world used to avoid allocating new worlds every frame when
/// swapping out the [`MainWorld`] for [`bevy::prelude::ExtractSchedule`].
#[derive(Resource, Default)]
struct ScratchMainWorld(World);

/// Executes the [`bevy::prelude::ExtractSchedule`] step of the renderer.
/// This updates the render world with the extracted ECS data of the current frame.
fn extract(main_world: &mut World, render_world: &mut World) {
    // temporarily add the app world to the render world as a resource
    let scratch_world = main_world.remove_resource::<ScratchMainWorld>().unwrap();
    let inserted_world = core::mem::replace(main_world, scratch_world.0);
    render_world.insert_resource(MainWorld(inserted_world));
    render_world.run_schedule(bevy::prelude::ExtractSchedule);

    // move the app world back, as if nothing happened.
    let inserted_world = render_world.remove_resource::<MainWorld>().unwrap();
    let scratch_world = core::mem::replace(main_world, *inserted_world);
    main_world.insert_resource(ScratchMainWorld(scratch_world));
}

unsafe fn initialize_render_app(app: &mut App) {
    app.init_resource::<ScratchMainWorld>();

    let mut render_app = SubApp::new();
    render_app.update_schedule = Some(Render.intern());

    let mut extract_schedule = Schedule::new(ExtractSchedule);
    // We skip applying any commands during the ExtractSchedule
    // so commands can be applied on the render thread.
    extract_schedule.set_build_settings(ScheduleBuildSettings {
        auto_insert_apply_deferred: false,
        ..default()
    });
    extract_schedule.set_apply_final_deferred(false);

    render_app
        .add_schedule(extract_schedule)
        .add_schedule(Render::base_schedule())
        .init_resource::<render_graph::RenderGraph>()
        .insert_resource(app.world().resource::<AssetServer>().clone())
        // .add_systems(ExtractSchedule, PipelineCache::extract_shaders)
        .add_systems(
            Render,
            (
                // This set applies the commands from the extract schedule while the render schedule
                // is running in parallel with the main app.
                apply_extract_commands.in_set(RenderSet::ExtractCommands),
                // (PipelineCache::process_pipeline_queue_system, render_system)
                //     .chain()
                //     .in_set(RenderSet::Render),
                despawn_temporary_render_entities.in_set(RenderSet::PostCleanup),
            ),
        );

    render_app.set_extract(|main_world, render_world| {
        {
            #[cfg(feature = "trace")]
            let _stage_span = tracing::info_span!("entity_sync").entered();
            entity_sync_system(main_world, render_world);
        }

        // run extract schedule
        extract(main_world, render_world);
    });

    let (sender, receiver) = bevy::time::create_time_channels();
    render_app.insert_resource(sender);
    app.insert_resource(receiver);
    app.insert_sub_app(RenderApp, render_app);
}

pub(crate) fn despawn_temporary_render_entities(
    world: &mut World,
    state: &mut SystemState<Query<Entity, With<TemporaryRenderEntity>>>,
    mut local: Local<Vec<Entity>>,
) {
    let query = state.get(world);

    local.extend(query.iter());

    // Ensure next frame allocation keeps order
    local.sort_unstable_by_key(|e| e.index());
    for e in local.drain(..).rev() {
        world.despawn(e);
    }
}

fn apply_extract_commands(render_world: &mut World) {
    render_world.resource_scope(|render_world, mut schedules: Mut<Schedules>| {
        schedules
            .get_mut(ExtractSchedule)
            .unwrap()
            .apply_deferred(render_world);
    });
}

impl Plugin for VulkanRenderPlugin {
    /// Initializes the renderer, sets up the [`RenderSet`] and creates the rendering sub-app.
    fn build(&self, app: &mut App) {
        app.init_asset::<Shader>()
            .init_asset_loader::<ShaderLoader>();

        let primary_window = app
            .world_mut()
            .query_filtered::<&RawHandleWrapperHolder, With<PrimaryWindow>>()
            .single(app.world())
            .ok()
            .cloned();
        let async_renderer = async move {
            let context = vulkano_util::context::VulkanoContext::new(VulkanoConfig::default());

            context
        };

        unsafe { initialize_render_app(app) };

        app.register_type::<SyncToRenderWorld>();
    }

    fn ready(&self, app: &App) -> bool {
        app.world()
            .get_resource::<FutureRenderResources>()
            .and_then(|frr| frr.0.try_lock().map(|locked| locked.is_some()).ok())
            .unwrap_or(true)
    }

    fn finish(&self, app: &mut App) {
        if let Some(future_render_resources) =
            app.world_mut().remove_resource::<FutureRenderResources>()
        {
            let RenderResources(device, queue, adapter_info, render_adapter, instance) =
                future_render_resources.0.lock().unwrap().take().unwrap();

            app.insert_resource(device.clone())
                .insert_resource(queue.clone())
                .insert_resource(adapter_info.clone())
                .insert_resource(render_adapter.clone());

            let render_app = app.sub_app_mut(RenderApp);

            render_app
                .insert_resource(instance)
                .insert_resource(PipelineCache::new(
                    device.clone(),
                    render_adapter.clone(),
                    self.synchronous_pipeline_compilation,
                ))
                .insert_resource(device)
                .insert_resource(queue)
                .insert_resource(render_adapter)
                .insert_resource(adapter_info);
        }
    }
}
