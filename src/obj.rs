use std::fs::File;
use std::io;
use std::io::BufRead;

use na::geometry::Point3;

use regex::Regex;

type Triangle = [Point3<f64>; 3];

pub fn parse_file(f: &File) -> io::Result<Vec<Triangle>> {
  let mut triangles = Vec::new();
  let mut vertices: Vec<Point3<f64>> = Vec::new();

  let vertex_rx = Regex::new(r"^v\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s*$").unwrap();
  let face_rx = Regex::new(r"^f\s+([0-9-]+)\s+([0-9-]+)\s+([0-9-]+)\s*$").unwrap();

  let reader = io::BufReader::new(f);
  for line in reader.lines() {
    let line = line?;
    match line.chars().next() {
      None => continue,
      Some('v') => {
        for captures in vertex_rx.captures_iter(&line) {
          vertices.push(Point3::new(captures[1].parse().unwrap(), captures[2].parse().unwrap(), captures[3].parse().unwrap()))
        }
      },
      Some('f') =>
        for captures in face_rx.captures_iter(&line) {
          triangles.push([
            vertices[captures[1].parse::<usize>().unwrap() - 1],
            vertices[captures[2].parse::<usize>().unwrap() - 1],
            vertices[captures[3].parse::<usize>().unwrap() - 1]
          ])
        },
      _ => {}
    }
  }
  Ok(triangles)
}
