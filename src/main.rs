extern crate nalgebra as na;
extern crate rand;
extern crate sfml;

use crate::sfml::graphics::RenderTarget;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::ops;
use sfml::graphics::{Image, RenderWindow, Sprite, Texture};
use sfml::window::{Event, Style};

fn main() {
  let width = 800;
  let height = 600;
  let mut window = RenderWindow::new((width, height), "Gleam", Style::CLOSE, &Default::default());

  let plane_dim = 0.9;

  let world = World {
    shapes: vec![
      (Shape::Sphere {
        center: Point::new(0.0, -1.0, 4.0),
        radius: 1.0
      }, Material {
        color: Color(Vector::new(1.0, 1.0, 1.0))
      }),
      (Shape::Sphere {
        center: Point::new(2.0, -1.0, 4.0),
        radius: 0.25
      }, Material {
        color: Color(Vector::new(0.5, 0.5, 0.5))
      }),
      (Shape::Plane {
        center: Point::new(0.0, -2.0, 0.0),
        normal: Normal::new_normalize(Vector::new(0.0, 1.0, 0.0))
      }, Material {
        color: Color(Vector::new(1.0, 0.3, 0.3) * plane_dim)
      }),
      (Shape::Plane {
        center: Point::new(-6.0, 0.0, 6.0),
        normal: Normal::new_normalize(Vector::new(1.0, 0.0, -1.0))
      }, Material {
        color: Color(Vector::new(0.3, 1.0, 0.3) * plane_dim)
      }),
      (Shape::Plane {
        center: Point::new(6.0, 0.0, 6.0),
        normal: Normal::new_normalize(Vector::new(-1.0, 0.0, -1.0))
      }, Material {
        color: Color(Vector::new(0.3, 0.3, 1.0) * plane_dim)
      })
    ],
    lights: vec![
      (Shape::Sphere {
        center: Point::new(0.0, 1.0, 3.0),
        radius: 0.1
      }, Light {
        color: Color(Vector::new(100.0, 100.0, 100.0))
      }),
      (Shape::Plane {
        center: Point::new(0.0, 5.0, 0.0),
        normal: Normal::new_normalize(Vector::new(0.0, -1.0, 0.0))
      }, Light {
        color: Color(Vector::new(1.0, 1.0, 1.0))
      })
    ]
  };

  let camera = Camera {
    ray: Ray {
      origin: Point::new(0.0, 0.0, 0.0),
      direction: Vector::new(0.0, 0.0, 1.0),
      color: Color(Vector::zeros())
    },
    fov: compute_fov((40.0 as f64).to_radians(), width as f64 / height as f64)
  };

  // let img = draw_viewport(&world, &camera, width, height);
  let mut img = Image::new(width, height);
  let mut y = 0;

  loop {
    while let Some(event) = window.poll_event() {
      match event {
        Event::Closed => return,
        _ => {}
      }
    }

    if y < height {
      draw_line(&world, &camera, &mut img, y, width, height);
      y += 1;
    }

    let tex = Texture::from_image(&img).expect("Couldn't create texture");
    window.draw(&Sprite::with_texture(&tex));

    window.display();
  }
}

fn compute_fov(fov_x: f64, ratio: f64) -> na::Vector2<f64> {
  let w = fov_x.tan();
  let h = w / ratio;
  let fov_y = h.atan();
  na::Vector2::new(fov_x, fov_y)
}

fn draw_line(world: &World, camera: &Camera, image: &mut Image, y: u32, w: u32, h: u32) {
  let yf = (y as i32 - h as i32 / 2) as f64 / ((h / 2) as f64);
  for x in 0 .. w {
    let xf = (x as i32 - w as i32 / 2) as f64 / ((w / 2) as f64);
    let mut c = Color(Vector::zeros());
    let sample_count = 1000;
    for _ in 0 .. sample_count {
      c += &cast_ray(world, camera, xf, -yf);
    }
    c /= sample_count as f64;
    image.set_pixel(x, y, sfml::graphics::Color::rgb(f64_to_u8(c.r()), f64_to_u8(c.g()), f64_to_u8(c.b())));
  }
}

fn draw_viewport(world: &World, camera: &Camera, w: u32, h: u32) -> Image {
  let mut image = Image::new(w, h);
  for x in 0 .. w {
    println!("x = {} / {}", x, w);
    for y in 0 .. h {
      let xf = (x as i32 - w as i32 / 2) as f64 / ((w / 2) as f64);
      let yf = (y as i32 - h as i32 / 2) as f64 / ((h / 2) as f64);
      let mut c = Color(Vector::zeros());
      let sample_count = 100;
      for _ in 0 .. sample_count {
        c += &cast_ray(world, camera, xf, -yf);
      }
      c /= sample_count as f64;
      image.set_pixel(x, y, sfml::graphics::Color::rgb(f64_to_u8(c.r()), f64_to_u8(c.g()), f64_to_u8(c.b())));
    }
  }
  image
}

fn f64_to_u8(f: f64) -> u8 {
  let f = f * 255.0;
  let f = if f >= 255.0 {
    255.0
  } else {
    f
  };
  f as u8
}

fn cast_ray(world: &World, camera: &Camera, x: f64, y: f64) -> Color {
  let plane_size = na::Vector2::new(camera.fov.x.tan(), camera.fov.y.tan());
  let mut ray = Ray {
    origin: camera.ray.origin,
    direction: Vector::new(x * plane_size.x, y * plane_size.y, 1.0).normalize(),
    color: Color(Vector::new(1.0, 1.0, 1.0))
  };
  for _ in 1..50 {
    let mut shape: Option<(f64, Point, &Shape, &Material)> = None;
    for (s, m) in &world.shapes {
      if let Some((d, p)) = s.intersect(&ray) {
        let swap = match shape {
          None => true,
          Some((old_distance, _, _, _)) => old_distance > d
        };
        if swap {
          shape = Some((d, p, s, m));
        }
      }
    }
    let mut light: Option<(f64, &Light)> = None;
    for (s, l) in &world.lights {
      if let Some((d, _)) = s.intersect(&ray) {
        let swap = match light {
          None => true,
          Some((old_distance, _)) => old_distance > d
        };
        if swap {
          light = Some((d, l));
        }
      }
    }
    match (shape, light) {
      (None, None) => {
        return Color(Vector::zeros())
      },
      (Some((_, p, s, m)), None) => {
        ray.reflect(&p, &s.normal(&p), m, &mut rand::thread_rng());
      }
      (None, Some((_, l))) => {
        ray.color *= &l.color;
        return ray.color;
      }
      (Some((ds, p, s, m)), Some((dl, l))) =>
        if ds < dl {
          ray.reflect(&p, &s.normal(&p), m, &mut rand::thread_rng());
        } else {
          ray.color *= &l.color;
          return ray.color;
        }
    }
  }

  Color(Vector::zeros())
}

type Point = na::Point3<f64>;
type Vector = na::Vector3<f64>;
type Normal = na::Unit<Vector>;

struct Color(Vector);

impl Color {

  fn r(&self) -> f64 {
    self.0.x
  }

  fn g(&self) -> f64 {
    self.0.y
  }

  fn b(&self) -> f64 {
    self.0.z
  }

}

impl ops::Mul<Color> for Color {

  type Output = Color;

  fn mul(self, color: Color) -> Color {
    Color(self.0.component_mul(&color.0))
  }

}

impl ops::MulAssign<&Color> for Color {

  fn mul_assign(&mut self, color: &Color) {
    self.0 = self.0.component_mul(&color.0)
  }
}

impl ops::DivAssign<f64> for Color {

  fn div_assign(&mut self, scalar: f64) {
    self.0 /= scalar;
  }

}

impl ops::AddAssign<&Color> for Color {

  fn add_assign(&mut self, color: &Color) {
    self.0 += color.0;
  }

}

struct Ray {
  origin: Point,
  direction: Vector,
  color: Color
}

struct Camera {
  ray: Ray,
  fov: na::Vector2<f64>
}

#[derive(Debug)]
enum Shape {
  Sphere {
    center: Point,
    radius: f64
  },
  Plane {
    center: Point,
    normal: Normal
  }
}

struct Material {
  color: Color
}

struct World {
  shapes: Vec<(Shape, Material)>,
  lights: Vec<(Shape, Light)>
}

struct Light {
  color: Color
}

impl Shape {

  fn intersect(&self, ray: &Ray) -> Option<(f64, na::Point3<f64>)> {
    match self {
      Shape::Sphere { center, radius } => {
        // |o + t * d - c|^2 = radius^2
        // |o-c + t * d| = radius
        // (oc.x + t * d.x)^2 + (oc.y + t * d.y)^2 + (oc.z + t * d.z)^2 = radius^2
        // oc.x^2 + 2 * t * d.x * oc.x + t^2 * d.x^2 + ... = radius^2
        // 2 * t * (d dot oc) + t^2 * |d|^2 = radius^2 - |oc|^2
        let oc = ray.origin - center;
        let a = ray.direction.norm_squared();
        let b = 2.0 * ray.direction.dot(&oc);
        let c = -(radius * radius - oc.norm_squared());
        let delta = b * b - 4.0 * a * c;
        if delta > 0.0 {
          let t1 = (-b - delta.sqrt()) / (2.0 * a);
          let t2 = (-b + delta.sqrt()) / (2.0 * a);
          let threshold = 0.001;
          let t = if t1 < t2 && t1 > threshold {
            t1
          } else {
            t2
          };
          if t <= threshold {
            None
          } else {
            let point = ray.origin + t * ray.direction;
            Some((t, point))
          }
        } else {
          None
        }
      },
      Shape::Plane { center, normal } => {
        // (c - p) · normal = 0
        // (c - (o + t d)) · normal = 0
        // (c.x - o.x - t d.x) * n.x + ... = 0
        // -t * (d dot n) = - co dot n
        // t = (c - o) dot n / d dot n
        let d_dot_n = ray.direction.dot(normal);
        if d_dot_n.abs() <= 0.01 {
          return None
        }
        let t = (center - ray.origin).dot(normal) / d_dot_n;
        if t <= 0.001 {
          return None
        }
        let point = ray.origin + t * ray.direction;
        if (center - point).dot(normal).abs() >= 0.001 {
          println!("wrong dot on plane intersection {}", (center - point).dot(normal));
        }
        Some((t, point))
      }
    }
  }

  fn normal(&self, point: &Point) -> Normal {
    match self {
      Shape::Sphere { center, .. } =>
        na::Unit::new_normalize(point - center),
      Shape::Plane { normal, .. } =>
        *normal,
    }
  }

}

impl Ray {

  fn reflect(&mut self, point: &Point, normal: &Normal, material: &Material, random: &mut ThreadRng) {
    let normal_part = (self.direction.dot(normal)) * normal.as_ref();
    let reflected_part = self.direction - normal_part;
    self.direction = -normal_part + reflected_part;
    self.direction += Vector::new(random.gen_range(-0.01, 0.01), random.gen_range(-0.01, 0.01), random.gen_range(-0.01, 0.01));
    self.color *= &material.color;
    self.origin = *point;
  }

}
