use super::{Triangle, Vertex};

use std::fs::File;
use std::io;
use std::io::BufRead;

use na::geometry::Point3;
use na::{Unit, Vector3};

pub fn parse_file(f: &File) -> io::Result<Vec<Triangle>> {
  let mut triangles = Vec::new();
  let mut vertices: Vec<Point3<f64>> = Vec::new();
  let mut normals: Vec<Unit<Vector3<f64>>> = Vec::new();

  let reader = io::BufReader::new(f);
  for line in reader.lines() {
    let line = line?;
    let parts: Vec<&str> = line.split(" ").collect();
    match parts[0] {
      "v" => {
        vertices.push(Point3::new(parts[1].parse().unwrap(), parts[2].parse().unwrap(), parts[3].parse().unwrap()))
      },
      "vn" => {
        normals.push(Unit::new_normalize(
          Vector3::new(
            parts[1].parse().unwrap(), parts[2].parse().unwrap(), parts[3].parse().unwrap()
          )
        ))
      },
      "f" =>
          triangles.push(Triangle::new([
          parse_vertex(parts[1], &vertices, &normals),
          parse_vertex(parts[2], &vertices, &normals),
          parse_vertex(parts[3], &vertices, &normals)
        ])),
      _ => {}
    }
  }
  Ok(triangles)
}

fn parse_vertex(src: &str, vertices: &Vec<Point3<f64>>, normals: &Vec<Unit<Vector3<f64>>>) -> Vertex {
  let parts: Vec<_> = src.split("/").collect();
  Vertex {
    position: vertices[parts[0].parse::<usize>().unwrap() - 1],
    normal: parts.get(2)
      .and_then(|p| p.parse::<usize>().ok())
      .map(|i| normals[i -  1])
  }
}
