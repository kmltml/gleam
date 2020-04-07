use super::{ Ray, Triangle, Intersection };

type Point = na::geometry::Point3<f64>;

#[derive(Copy, Clone, Debug)]
struct Bounds {
  min: Point,
  max: Point
}

#[derive(Debug)]
pub struct KDTree {
  root: Node,
  bounds: Bounds,
  elements: Vec<Triangle>
}

#[derive(Copy, Clone, Debug)]
enum Axis {
  X, Y, Z
}

#[derive(Debug)]
enum Node {
  Partition {
    axis: Axis,
    division: f64,
    left: Box<Node>,
    right: Box<Node>,
  },
  Leaf {
    children: Vec<usize>
  }
}

impl Axis {

  fn axes() -> Vec<Axis> {
    vec![Axis::X, Axis::Y, Axis::Z]
  }

  fn get(&self, p: &Point) -> f64 {
    match self {
      Axis::X => p.x,
      Axis::Y => p.y,
      Axis::Z => p.z
    }
  }

  fn set(&self, p: &mut Point, v: f64) {
    match self {
      Axis::X => p.x = v,
      Axis::Y => p.y = v,
      Axis::Z => p.z = v
    }
  }

  fn get_vec(&self, p: &na::Vector3<f64>) -> f64 {
    match self {
      Axis::X => p.x,
      Axis::Y => p.y,
      Axis::Z => p.z
    }
  }

}

impl Bounds {

  fn for_points(points: &[Point]) -> Bounds {
    let mut min = points[0];
    let mut max = points[0];
    for p in points.iter().skip(1) {
      min.x = min.x.min(p.x);
      min.y = min.y.min(p.y);
      min.z = min.z.min(p.z);

      max.x = max.x.max(p.x);
      max.y = max.y.max(p.y);
      max.z = max.z.max(p.z);
    }
    Bounds { min, max }
  }

  fn for_triangles(triangles: &[Triangle]) -> Bounds {
    let mut ret = Bounds::for_points(&triangles[0].vertices);
    for t in triangles.iter().skip(1) {
      ret = ret.union(&Bounds::for_points(&t.vertices))
    }
    ret
  }

  fn contains(&self, p: &Point) -> bool {
    p.x >= self.min.x && p.x <= self.max.x &&
    p.y >= self.min.y && p.y <= self.max.y &&
    p.z >= self.min.z && p.z <= self.max.z
  }

  fn intersect(&self, ray: &Ray) -> Option<f64> {
    if self.contains(&ray.origin) {
      return Some(0.0)
    }
    // println!("{:?} , {:?}", self, ray);
    let mut min_t: Option<f64> = None;
    for axis in Axis::axes() {
      for bound in &[self.min, self.max] {
        let b = axis.get(bound);
        let x0 = axis.get(&ray.origin);
        let dx = axis.get_vec(&ray.direction);
        // b = x0 + t * dx
        // t = (b - x0) / dx
        let t = (b - x0) / dx;
        if t < 0.001 {
          continue
        }
        let mut p = ray.origin + ray.direction * t;
        axis.set(&mut p, (axis.get(&self.min) + axis.get(&self.max)) / 2.0);
        if self.contains(&p) {
          let swap = match min_t {
            None => true,
            Some(previous_t) => t < previous_t
          };
          if swap {
            min_t = Some(t);
          }
        }
      }
    }
    min_t
  }

  fn intersects_bounds(&self, other: &Bounds) -> bool {
    for axis in Axis::axes() {
      let min_a = axis.get(&self.min);
      let max_a = axis.get(&self.max);
      let min_b = axis.get(&other.min);
      let max_b = axis.get(&other.max);
      if min_a > max_b || max_a < min_b {
        return false;
      }
    }
    true
  }

  fn union(&self, other: &Bounds) -> Bounds {
    Bounds {
      min: Point::new(
        self.min.x.min(other.min.x),
        self.min.y.min(other.min.y),
        self.min.z.min(other.min.z),
      ),
      max: Point::new(
        self.max.x.max(other.max.x),
        self.max.y.max(other.max.y),
        self.max.z.max(other.max.z),
      )
    }
  }

  fn split(&self, axis: Axis, split_location: f64) -> (Bounds, Bounds) {
    let mut left_max = self.max;
    let mut right_min = self.min;
    axis.set(&mut left_max, split_location);
    axis.set(&mut right_min, split_location);
    (Bounds {
      min: self.min, max: left_max
    }, Bounds {
      min: right_min, max: self.max
    })
  }

}

impl Node {

  pub fn print_structure(&self) {
    match self {
      Node::Leaf { children } => print!("{}", children.len()),
      Node::Partition { ref left, ref right, .. } => {
        print!("(");
        left.print_structure();
        print!(" ");
        right.print_structure();
        print!(")");
      }
    }
  }

}

impl KDTree {

  pub fn build(triangles: &[Triangle]) -> KDTree {
    let bounds = Bounds::for_triangles(triangles);
    let elements = triangles.iter().copied().collect();

    fn partition(triangles: Vec<usize>, elements: &Vec<Triangle>, bounds: Bounds, axis: Axis) -> Node {
      if triangles.len() <= 4 {
        return Node::Leaf {
          children: triangles
        };
      }
      let split_point = (axis.get(&bounds.min) + axis.get(&bounds.max)) / 2.0;
      let (left_bound, right_bound) = bounds.split(axis, split_point);
      let left_triangles: Vec<usize> = triangles.iter()
        .map(|t| *t)
        .filter(|t| {
          Bounds::for_triangles(&[elements[*t]])
            .intersects_bounds(&left_bound)
        })
        .collect();
      let right_triangles: Vec<usize> = triangles.iter()
        .map(|t| *t)
        .filter(|t| {
          Bounds::for_triangles(&[elements[*t]])
            .intersects_bounds(&right_bound)
        })
        .collect();

      if left_triangles.is_empty() ||
        right_triangles.is_empty() ||
        left_triangles.len() == triangles.len() ||
        right_triangles.len() == triangles.len() {

        return Node::Leaf {
          children: triangles
        };
      }

      let next_axis = match axis {
        Axis::X => Axis::Y,
        Axis::Y => Axis::Z,
        Axis::Z => Axis::X
      };

      Node::Partition {
        axis,
        division: split_point,
        left: Box::new(partition(left_triangles, elements, left_bound, next_axis)),
        right: Box::new(partition(right_triangles, elements, right_bound, next_axis))
      }
    }

    let all_triangles: Vec<usize> = triangles.iter()
      .enumerate()
      .map(|(i, _)| i)
      .collect();
    let root = partition(all_triangles, &elements, bounds, Axis::X);
    KDTree {
      root, bounds, elements
    }
  }

  pub fn print_structure(&self) {
    self.root.print_structure()
  }

  pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {

    fn rec(node: &Node, bounds: &Bounds, ray: &Ray, elements: &Vec<Triangle>) -> Option<Intersection> {
      match *node {
        Node::Leaf { ref children } => {
          children.iter()
            .filter_map(|i| elements[*i].intersect(ray))
            .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
        },
        Node::Partition { axis, division, ref left, ref right } => {
          let (left_bounds, right_bounds) = bounds.split(axis, division);
          let mut v: Vec<(f64, Bounds, &Box<Node>)> = vec![];
          if let Some(t) = left_bounds.intersect(ray) {
            v.push((t, left_bounds, left));
          }
          if let Some(t) = right_bounds.intersect(ray) {
            v.push((t, right_bounds, right));
          }
          v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
          for (_, b, n) in v {
            if let i @ Some(_) = rec(n, &b, ray, elements) {
              return i;
            }
          }
          return None
        }
      }
    }

    if let None = self.bounds.intersect(ray) {
      return None;
    }
    rec(&self.root, &self.bounds, ray, &self.elements)
  }

}
