mod bvh;
mod camera;
mod hittable;
mod interval;
mod material;
mod ray;
mod scene;
mod sphere;
mod vec3;

use std::process::ExitCode;
use std::time::Instant;

use camera::{Camera, CameraConfig};
use scene::SCENE_NAMES;

const USAGE: &str = "\
raytracker — a path tracer

usage: raytracker [options]

options:
  --scene <name>     scene to render: three | glass | final  (default: three)
  --width <px>       image width in pixels                   (default: 400)
  --samples <n>      rays per pixel; higher is less noisy    (default: 100)
  --depth <n>        maximum bounces per ray                 (default: 50)
  --seed <n>         sampling seed; changes the noise only   (default: 0)
  --out <path>       output PNG path                         (default: render.png)
  --no-bvh           skip the BVH and test every object      (slow; for timing)
  -h, --help         show this message

examples:
  raytracker --scene three --width 400 --samples 50 --out preview.png
  raytracker --scene final --width 1200 --samples 500 --out render.png
";

struct Options {
    scene: String,
    out: String,
    config: CameraConfig,
    use_bvh: bool,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let options = match parse_args(&args) {
        Ok(Some(options)) => options,
        // --help: not an error.
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("error: {message}\n");
            eprint!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    let Some(scene) = scene::build(&options.scene, options.config) else {
        eprintln!(
            "error: unknown scene '{}' (expected one of: {})",
            options.scene,
            SCENE_NAMES.join(", ")
        );
        return ExitCode::FAILURE;
    };

    if scene.world.is_empty() {
        eprintln!("error: scene '{}' contains no objects", options.scene);
        return ExitCode::FAILURE;
    }

    let camera = Camera::new(scene.camera);
    let object_count = scene.world.len();

    // Flattening the object list into a BVH turns the per-ray cost from linear
    // in the object count into roughly logarithmic. Worth about 4.5x on the
    // 485-object final scene; the gap widens as scenes grow. Compare the two
    // with --no-bvh.
    let build_start = Instant::now();
    let world: Box<dyn hittable::Hittable> = if options.use_bvh {
        Box::new(bvh::BvhNode::from_list(scene.world))
    } else {
        Box::new(scene.world)
    };
    let build_time = build_start.elapsed();

    if options.use_bvh {
        eprintln!(
            "scene '{}': {} objects, BVH built in {:.1?}",
            options.scene, object_count, build_time
        );
    } else {
        eprintln!(
            "scene '{}': {} objects, no BVH (linear search)",
            options.scene, object_count
        );
    }
    eprintln!(
        "rendering {}x{} at {} spp, max depth {}",
        camera.image_width(),
        camera.image_height(),
        scene.camera.samples_per_pixel,
        scene.camera.max_depth
    );

    let render_start = Instant::now();
    let pixels = camera.render(world.as_ref());
    let render_time = render_start.elapsed();

    match image::save_buffer(
        &options.out,
        &pixels,
        camera.image_width(),
        camera.image_height(),
        image::ColorType::Rgb8,
    ) {
        Ok(()) => {
            eprintln!("wrote {} in {:.1?}", options.out, render_time);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: could not write {}: {e}", options.out);
            ExitCode::FAILURE
        }
    }
}

/// Returns `Ok(None)` when the user asked for help.
fn parse_args(args: &[String]) -> Result<Option<Options>, String> {
    let mut options = Options {
        scene: "three".to_string(),
        out: "render.png".to_string(),
        config: CameraConfig::default(),
        use_bvh: true,
    };

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        // Every flag below takes exactly one value, so a missing one is an error
        // rather than a silent default.
        let mut value = || {
            iter.next()
                .ok_or_else(|| format!("{arg} needs a value"))
                .cloned()
        };

        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--no-bvh" => options.use_bvh = false,
            "--scene" => options.scene = value()?,
            "--out" | "-o" => options.out = value()?,
            "--width" => options.config.image_width = parse_number(arg, &value()?)?,
            "--samples" => options.config.samples_per_pixel = parse_number(arg, &value()?)?,
            "--depth" => options.config.max_depth = parse_number(arg, &value()?)?,
            "--seed" => options.config.seed = parse_number(arg, &value()?)?,
            other => return Err(format!("unrecognized argument '{other}'")),
        }
    }

    if options.config.image_width == 0 {
        return Err("--width must be at least 1".to_string());
    }
    if options.config.samples_per_pixel == 0 {
        return Err("--samples must be at least 1".to_string());
    }
    if options.config.max_depth == 0 {
        return Err("--depth must be at least 1".to_string());
    }

    Ok(Some(options))
}

fn parse_number<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, String> {
    raw.parse()
        .map_err(|_| format!("{flag} expects a number, got '{raw}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults_apply_when_no_flags_are_given() {
        let options = parse_args(&[]).unwrap().unwrap();
        assert_eq!(options.scene, "three");
        assert_eq!(options.out, "render.png");
        assert_eq!(options.config.image_width, 400);
    }

    #[test]
    fn flags_override_defaults() {
        let options = parse_args(&args(&[
            "--scene", "final", "--width", "800", "--samples", "10", "--depth", "5", "--seed",
            "99", "--out", "x.png",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(options.scene, "final");
        assert_eq!(options.config.image_width, 800);
        assert_eq!(options.config.samples_per_pixel, 10);
        assert_eq!(options.config.max_depth, 5);
        assert_eq!(options.config.seed, 99);
        assert_eq!(options.out, "x.png");
    }

    #[test]
    fn bvh_is_on_unless_disabled() {
        assert!(parse_args(&[]).unwrap().unwrap().use_bvh);
        assert!(!parse_args(&args(&["--no-bvh"])).unwrap().unwrap().use_bvh);
    }

    #[test]
    fn help_short_circuits() {
        assert!(parse_args(&args(&["--help"])).unwrap().is_none());
        assert!(parse_args(&args(&["-h"])).unwrap().is_none());
    }

    #[test]
    fn bad_input_is_reported_not_ignored() {
        assert!(parse_args(&args(&["--width"])).is_err());
        assert!(parse_args(&args(&["--width", "abc"])).is_err());
        assert!(parse_args(&args(&["--width", "0"])).is_err());
        assert!(parse_args(&args(&["--samples", "0"])).is_err());
        assert!(parse_args(&args(&["--depth", "0"])).is_err());
        assert!(parse_args(&args(&["--nonsense"])).is_err());
    }
}
