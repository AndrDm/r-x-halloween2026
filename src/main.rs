//! Halloween 2026 — industrial C-arm X-ray pumpkin inspector.
//!
//! Single-file eframe/egui app that runs both on desktop (native) and in the
//! browser (trunk / wasm32, `trunk build --release --public-url . --no-default-features`).
//!
//! The X-ray image is computed on the CPU every frame: rays are cast from the
//! source (left) towards the detector plane (right) through the *static*
//! pumpkin mesh. The model rotation is applied to the rays instead, keeping
//! the uniform-voxel-grid acceleration structure valid across frames.
//! Mesh hits use the Möller–Trumbore intersection algorithm; the in-material
//! chord length drives intensity: air = maximal (bright), material = darker
//! (exponential attenuation). Seeds inside the hollow cavity are analytic
//! axis-aligned quadrics integrated along the same rays.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::f32::consts::{PI, TAU};
// `std::time::Instant` panics on wasm32-unknown-unknown ("time not
// implemented on this platform"), so on web we wrap the monotonic
// `performance.now()` clock behind the same tiny API used by the app.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
struct Instant {
    ms: f64,
}

#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        Self {
            ms: eframe::web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0),
        }
    }

    fn duration_since(&self, earlier: Instant) -> std::time::Duration {
        std::time::Duration::from_secs_f32(((self.ms - earlier.ms).max(0.0) * 1e-3) as f32)
    }

    fn elapsed(&self) -> std::time::Duration {
        Self::now().duration_since(*self)
    }
}

#[cfg(target_arch = "wasm32")]
impl std::ops::Sub for Instant {
    type Output = std::time::Duration;

    fn sub(self, rhs: Instant) -> std::time::Duration {
        self.duration_since(rhs)
    }
}

use eframe::egui;
use egui::{
    Align2, Color32, ColorImage, FontId, Sense, TextureHandle, TextureOptions, Vec2, pos2, vec2,
};

// ---------------------------------------------------------------------------
// palette
// ---------------------------------------------------------------------------

const PANEL_BG: Color32 = Color32::from_rgb(16, 17, 19);
const ACCENT: Color32 = Color32::from_rgb(255, 140, 0);
const ACCENT_DIM: Color32 = Color32::from_rgb(178, 96, 12);
const TEXT_DIM: Color32 = Color32::from_rgb(145, 145, 152);

// years: attenuation coefficients (per unit pumpkin radius ray length)
const MU_FLESH_BASE: f32 = 1.35;
const MU_SEED: f32 = 7.0;
const MU_STEM: f32 = 2.0;

// source / detector geometry (1 unit = 10 cm of world space)
const SRC_X: f32 = -4.5; // source tube position
const DET_X: f32 = 4.5; // detector plane position
const DET_HALF: f32 = 2.5; // detector half-size on the plane
const DETECTOR_MM: f32 = 27.0; // physical active area of the CMOS panel, mm

/// Pixel pitch of the CMOS panel for a given matrix size, in micrometers.
fn det_pitch_um(res: usize) -> u32 {
    (DETECTOR_MM * 1000.0 / res as f32).round() as u32
}

// ---------------------------------------------------------------------------
// small 3d math
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

impl V3 {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }
    fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    fn normalized(self) -> Self {
        let l = self.len();
        if l > 1e-9 {
            self.scale(1.0 / l)
        } else {
            Self::new(0.0, 0.0, 0.0)
        }
    }
    fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    fn min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    fn max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    /// rotate about Y by angle a (right-handed)
    fn rot_y(self, c: f32, s: f32) -> Self {
        Self::new(self.x * c + self.z * s, self.y, -self.x * s + self.z * c)
    }
    /// inverse of rot_y
    fn rot_y_inv(self, c: f32, s: f32) -> Self {
        Self::new(self.x * c - self.z * s, self.y, self.x * s + self.z * c)
    }
    /// rotate about X by angle a
    fn rot_x(self, c: f32, s: f32) -> Self {
        Self::new(self.x, self.y * c - self.z * s, self.y * s + self.z * c)
    }
}

impl std::ops::Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl std::ops::Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

// ---------------------------------------------------------------------------
// mesh + möller–trumbore
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Tri {
    a: V3,
    b: V3,
    c: V3,
    group: u8, // 0 = body surface, 1 = cavity surface, 2 = stem
}

/// Möller–Trumbore, double-sided.
fn mt(o: V3, d: V3, t: &Tri) -> Option<f32> {
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let h = d.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-10 {
        return None;
    }
    let f = 1.0 / det;
    let s = o - t.a;
    let u = f * s.dot(h);
    if !(-1e-4..=1.0001).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * d.dot(q);
    if !(-1e-4..=1.0001).contains(&v) {
        return None;
    }
    let t_hit = f * e2.dot(q);
    if t_hit > 1e-4 {
        Some(t_hit)
    } else {
        None
    }
}

fn ray_box(o: V3, d: V3, lo: V3, hi: V3) -> Option<(f32, f32)> {
    let mut t0 = 0.0f32;
    let mut t1 = f32::INFINITY;
    for axis in 0..3 {
        let (o, d, l, h) = match axis {
            0 => (o.x, d.x, lo.x, hi.x),
            1 => (o.y, d.y, lo.y, hi.y),
            _ => (o.z, d.z, lo.z, hi.z),
        };
        if d.abs() < 1e-9 {
            if o < l || o > h {
                return None;
            }
        } else {
            let mut a = (l - o) / d;
            let mut b = (h - o) / d;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t0 = t0.max(a);
            t1 = t1.min(b);
        }
    }
    if t1 > t0 {
        Some((t0, t1))
    } else {
        None
    }
}

/// Uniform voxel grid over the static mesh, built once at startup.
struct Grid {
    lo: V3,
    cell: f32,
    n: i32,
    cells: Vec<Vec<u32>>,
    tris: Vec<Tri>,
}

impl Grid {
    fn build(tris: Vec<Tri>, n: i32, extent: f32) -> Self {
        let lo = V3::splat(-extent);
        let hi = V3::splat(extent);
        let cell = (hi.x - lo.x) / n as f32;
        let mut cells = vec![Vec::new(); (n * n * n) as usize];
        let idx_of = |p: V3| -> (i32, i32, i32) {
            let f = |v: f32| -> i32 {
                ((v - lo.x) / cell).floor().clamp(0.0, (n - 1) as f32) as i32
            };
            (f(p.x), f(p.y), f(p.z))
        };
        for (ti, t) in tris.iter().enumerate() {
            // conservative eps expansion so triangles hugging cell boundaries
            // are registered in the neighboring cells too
            let eps = V3::splat(cell * 0.01);
            let (x0, y0, z0) = idx_of(t.a.min(t.b).min(t.c) - eps);
            let (x1, y1, z1) = idx_of(t.a.max(t.b).max(t.c) + eps);
            let ti = ti as u32;
            for ix in x0..=x1 {
                for iy in y0..=y1 {
                    for iz in z0..=z1 {
                        let cell_i = (ix * n + iy) * n + iz;
                        cells[cell_i as usize].push(ti);
                    }
                }
            }
        }
        Self {
            lo,
            cell,
            n,
            cells,
            tris,
        }
    }

    /// Amanatides & Woo voxel traversal; appends (group, t) hits.
    fn trace(
        &self,
        o: V3,
        d: V3,
        stamp: u32,
        tested: &mut [u32],
        out: &mut Vec<(u8, f32)>,
    ) {
        let Some((tb0, tb1)) = ray_box(o, d, self.lo, V3::splat(-self.lo.x)) else {
            return;
        };
        if tb1 <= 0.0 {
            return;
        }
        let t_cur = tb0.max(0.0) + 1e-4;
        let p = o + d.scale(t_cur);
        let inv = 1.0 / self.cell;
        let cl = |x: f32| -> i32 { x.floor().clamp(0.0, (self.n - 1) as f32) as i32 };
        let mut vx = cl((p.x - self.lo.x) * inv);
        let mut vy = cl((p.y - self.lo.y) * inv);
        let mut vz = cl((p.z - self.lo.z) * inv);
        let steps = V3::new(
            if d.x > 0.0 { 1.0 } else { -1.0 },
            if d.y > 0.0 { 1.0 } else { -1.0 },
            if d.z > 0.0 { 1.0 } else { -1.0 },
        );
        let boundary = |i: i32, o: f32, d: f32| -> f32 {
            if d.abs() < 1e-9 {
                f32::INFINITY
            } else {
                let b = if d > 0.0 { (i + 1) as f32 } else { i as f32 };
                let dist = self.lo.x + b * self.cell - o;
                (dist / d).max(0.0)
            }
        };
        let mut t_next = V3::new(
            boundary(vx, p.x, d.x),
            boundary(vy, p.y, d.y),
            boundary(vz, p.z, d.z),
        );
            let t_delta = V3::new(
                if d.x.abs() < 1e-9 { f32::INFINITY } else { self.cell / d.x.abs() },
                if d.y.abs() < 1e-9 { f32::INFINITY } else { self.cell / d.y.abs() },
                if d.z.abs() < 1e-9 { f32::INFINITY } else { self.cell / d.z.abs() },
            );

        for _ in 0..512 {
            let cell_i = ((vx * self.n + vy) * self.n + vz) as usize;
            for &ti in &self.cells[cell_i] {
                let iu = ti as usize;
                if tested[iu] != stamp {
                    tested[iu] = stamp;
                    if let Some(t_hit) = mt(o, d, &self.tris[iu]) {
                        if t_hit >= tb0 && t_hit <= tb1 {
                            out.push((self.tris[iu].group, t_hit));
                        }
                    }
                }
            }
            // advance to next voxel
            let (axis, t_exit) = if t_next.x <= t_next.y && t_next.x <= t_next.z {
                (0, t_next.x)
            } else if t_next.y <= t_next.z {
                (1, t_next.y)
            } else {
                (2, t_next.z)
            };
            if t_exit > tb1 {
                break;
            }
            match axis {
                0 => {
                    vx += steps.x as i32;
                    t_next.x += t_delta.x;
                }
                1 => {
                    vy += steps.y as i32;
                    t_next.y += t_delta.y;
                }
                _ => {
                    vz += steps.z as i32;
                    t_next.z += t_delta.z;
                }
            }
            if !(0..self.n).contains(&vx)
                || !(0..self.n).contains(&vy)
                || !(0..self.n).contains(&vz)
            {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// geometry generation (done once)
// ---------------------------------------------------------------------------

/// Ribbed radial profile of the pumpkin skin.
fn ribs(theta: f32) -> f32 {
    1.0 + 0.045 * (9.0 * theta).cos() + 0.012 * (18.0 * theta).cos()
}

fn sphere_pt(theta: f32, phi: f32, rs: f32, hs: f32) -> V3 {
    let s = phi.sin().max(1e-5);
    let rad = 0.92 * rs * ribs(theta) * s;
    V3::new(rad * theta.cos(), 0.80 * hs * phi.cos(), rad * theta.sin())
}

/// Parametric closed surface (lathe over theta x phi).
fn lathe(tth: usize, tph: usize, rs: f32, hs: f32, group: u8, out: &mut Vec<Tri>) {
    for i in 0..tth {
        for j in 0..tph {
            let t0 = TAU * i as f32 / tth as f32;
            let t1 = TAU * (i + 1) as f32 / tth as f32;
            let p0 = PI * j as f32 / tph as f32;
            let p1 = PI * (j + 1) as f32 / tph as f32;
            let a = sphere_pt(t0, p0, rs, hs);
            let b = sphere_pt(t1, p0, rs, hs);
            let c = sphere_pt(t1, p1, rs, hs);
            let dd = sphere_pt(t0, p1, rs, hs);
            out.push(Tri { a, b, c, group });
            out.push(Tri { a, b: c, c: dd, group });
        }
    }
}

/// Tapered, slightly leaning stem with closed caps.
fn stem(out: &mut Vec<Tri>) {
    const RINGS: usize = 5;
    const SEC: usize = 12;
    let axis = |t: f32| V3::new(0.05 + 0.10 * t * t, 0.70 + 0.44 * t, 0.0);
    let rad = |t: f32| 0.13 - 0.08 * t;
    let ring_v = |t: f32, k: usize| -> V3 {
        let ang = TAU * k as f32 / SEC as f32;
        let r = rad(t);
        let c = axis(t);
        V3::new(c.x + r * ang.cos(), c.y, c.z + r * ang.sin())
    };
    for r in 0..RINGS - 1 {
        let t0 = r as f32 / (RINGS - 1) as f32;
        let t1 = (r + 1) as f32 / (RINGS - 1) as f32;
        for k in 0..SEC {
            let a = ring_v(t0, k);
            let b = ring_v(t0, (k + 1) % SEC);
            let c = ring_v(t1, k);
            let dd = ring_v(t1, (k + 1) % SEC);
            out.push(Tri { a, b, c, group: 2 });
            out.push(Tri { a: b, b: dd, c, group: 2 });
        }
    }
    // bottom cap (hidden inside the pumpkin shell)
    let c0 = axis(0.0);
    for k in 0..SEC {
        out.push(Tri {
            a: c0,
            b: ring_v(0.0, k),
            c: ring_v(0.0, (k + 1) % SEC),
            group: 2,
        });
    }
    // top cap
    let c1 = axis(1.0) + V3::new(0.0, 0.02, 0.0);
    for k in 0..SEC {
        out.push(Tri {
            a: c1,
            b: ring_v(1.0, (k + 1) % SEC),
            c: ring_v(1.0, k),
            group: 2,
        });
    }
}

/// Axis-aligned ellipsoid seeds floating in the hollow cavity (analytic).
fn seeds() -> Vec<(V3, V3)> {
    let rnd = |i: f32, s: f32| -> f32 {
        (i * 12.9898 + s * 78.233).sin() * 43758.5453
    };
    let frac = |v: f32| -> f32 { v - v.floor() };
    let mut out = Vec::new();
    for i in 0..22 {
        let i = i as f32;
        let th = TAU * (i * 0.618034);
        let r = 0.30 + 0.24 * frac(rnd(i, 1.0)).abs();
        let y = -0.28 + 0.26 * frac(rnd(i, 2.0)).abs();
        let center = V3::new(th.cos() * r, y, th.sin() * r);
        let radii = V3::new(
            0.12 + 0.05 * frac(rnd(i, 3.0)).abs(),
            0.045 + 0.02 * frac(rnd(i, 4.0)).abs(),
            0.07 + 0.03 * frac(rnd(i, 5.0)).abs(),
        );
        out.push((center, radii));
    }
    out
}

// ---------------------------------------------------------------------------
// jack-o'-lantern carvings
// ---------------------------------------------------------------------------

/// Convex carved opening through the near-side shell: triangle prism, anchored
/// on the body surface, extruded along the local normal (n > 0 = outward).
struct Carving {
    anchor: V3, // center point on the body surface
    n: V3,      // approx surface normal (radial)
    u: V3,      // horizontal tangent
    v: V3,      // vertical tangent
    tri: [(f32, f32); 3], // CCW triangle in (u, v) plane
    n_out: f32, // extends a bit outside the surface
    n_in: f32,  // carving depth into the flesh
}

/// A point on the body surface, given a direction (unit-ish) from the origin.
fn surface_point(dir: V3) -> V3 {
    let d = dir.normalized();
    let theta = d.z.atan2(d.x);
    let cf = (d.y / 0.80).clamp(-0.999, 0.999);
    let sp = (1.0 - cf * cf).max(1e-4).sqrt();
    let rad = 0.92 * ribs(theta) * sp;
    V3::new(rad * theta.cos(), 0.80 * cf, rad * theta.sin())
}

/// Build a carving at a given surface direction. The model "front" faces the
/// X-ray source (-X) so the carved features read crisply; `az` offsets the
/// direction along +Z. `tri` is placed in the surface tangent plane.
fn make_carving(y: f32, az: f32, tri: [(f32, f32); 3], n_in: f32) -> Carving {
    let h = (1.0 - y * y).max(0.0).sqrt();
    let anchor = surface_point(V3::new(-h * az.cos(), y, h * az.sin()));
    let n = anchor.normalized();
    let mut u = V3::new(0.0, 1.0, 0.0).cross(n).normalized();
    if u.len() < 0.5 {
        u = V3::new(1.0, 0.0, 0.0);
    }
    let v = n.cross(u); // v points up (world +Y)
    let mut tri = tri;
    // ensure CCW winding so "inside" is to the left of every edge
    let cz = (tri[1].0 - tri[0].0) * (tri[2].1 - tri[0].1)
        - (tri[1].1 - tri[0].1) * (tri[2].0 - tri[0].0);
    if cz < 0.0 {
        tri.swap(1, 2);
    }
    Carving {
        anchor,
        n,
        u,
        v,
        tri,
        n_out: 0.04,
        n_in,
    }
}

/// The classic face: two triangular eyes, a small nose, a wide grin.
fn carvings() -> Vec<Carving> {
    let mut out = Vec::new();
    // eyes
    out.push(make_carving(
        0.18,
        -0.38,
        [(-0.16, -0.06), (0.0, 0.11), (0.16, -0.06)],
        0.22,
    ));
    out.push(make_carving(
        0.18,
        0.38,
        [(-0.16, -0.06), (0.0, 0.11), (0.16, -0.06)],
        0.22,
    ));
    // nose
    out.push(make_carving(
        0.02,
        0.0,
        [(-0.07, 0.045), (0.07, 0.045), (0.0, -0.08)],
        0.20,
    ));
    // grin: overlapping triangles sweep left -> right, carved deepest at center
    let n = 11;
    for k in 0..n {
        let f = k as f32 / (n - 1) as f32;
        let az = -0.68 + 1.36 * f;
        let depth = 0.10 + 0.14 * (1.0 - (2.0 * f - 1.0).abs().powi(2));
        // alternate up/down triangles make the zigzag teeth
        let sgn = if k % 2 == 0 { 1.0 } else { -1.0 };
        out.push(make_carving(
            -0.22 - 0.11 * (2.0 * f - 1.0).abs(),
            az,
            [(-0.09, sgn * 0.07), (0.09, sgn * 0.07), (0.0, -sgn * 0.08)],
            depth,
        ));
    }
    out
}

/// Ray interval inside a carving prism, in ray parametrization from `o`.
/// Returns (t_lo, t_hi) with t_lo < t_hi, or None if the ray misses it.
fn carve_interval(o: V3, d: V3, c: &Carving) -> Option<(f32, f32)> {
    let rel = o - c.anchor;
    let o_n = rel.dot(c.n);
    let d_n = d.dot(c.n);
    let o_u = rel.dot(c.u);
    let d_u = d.dot(c.u);
    let o_v = rel.dot(c.v);
    let d_v = d.dot(c.v);
    let mut t_lo = f32::NEG_INFINITY;
    let mut t_hi = f32::INFINITY;
    // normal slab: n in [-n_in, +n_out]
    {
        let (nl, nh) = (-c.n_in, c.n_out);
        if d_n.abs() < 1e-9 {
            if o_n < nl || o_n > nh {
                return None;
            }
        } else {
            let mut a = (nl - o_n) / d_n;
            let mut b = (nh - o_n) / d_n;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t_lo = t_lo.max(a);
            t_hi = t_hi.min(b);
        }
    }
    // triangle edges: inside = left of every CCW edge
    for i in 0..3 {
        let (u0, v0) = c.tri[i];
        let (u1, v1) = c.tri[(i + 1) % 3];
        let (w0, w1) = (-(v1 - v0), u1 - u0); // inward (left) normal
        let c0 = w0 * (o_u - u0) + w1 * (o_v - v0);
        let cd = w0 * d_u + w1 * d_v;
        if cd.abs() < 1e-12 {
            if c0 < 0.0 {
                return None;
            }
        } else {
            let mut a = -c0 / cd;
            let mut b = f32::INFINITY;
            if cd < 0.0 {
                b = a;
                a = f32::NEG_INFINITY;
            }
            t_lo = t_lo.max(a);
            t_hi = t_hi.min(b);
        }
    }
    if t_hi > t_lo {
        Some((t_lo, t_hi))
    } else {
        None
    }
}

// chord length inside an axis-aligned ellipsoid for a normalized ray
fn ellipsoid_chord(o: V3, d: V3, center: V3, radii: V3) -> f32 {
    let rel = o - center;
    let ir = V3::new(1.0 / radii.x, 1.0 / radii.y, 1.0 / radii.z);
    let a = d.scale(ir.x).x * d.scale(ir.x).x
        + d.scale(ir.y).y * d.scale(ir.y).y
        + d.scale(ir.z).z * d.scale(ir.z).z;
    let rel_i = V3::new(rel.x * ir.x, rel.y * ir.y, rel.z * ir.z);
    let b = 2.0 * rel_i.dot(d);
    let c = rel_i.dot(rel_i) - 1.0;
    let disc = b * b - 4.0 * a * c;
    if disc <= 0.0 {
        return 0.0;
    }
    let s = disc.sqrt();
    (s - b) / a // distance between roots along the *normalized* ray
}

/// Sort + merge shared-edge duplicate hits, keeping true crossings 1e-3 apart.
fn dedupe_hits(ts: &mut Vec<f32>) {
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut uniq: Vec<f32> = Vec::with_capacity(ts.len());
    for &t in ts.iter() {
        match uniq.last() {
            Some(&last) if (t - last) < 1e-3 => {}
            _ => uniq.push(t),
        }
    }
    *ts = uniq;
}

/// Chord through a (star-shaped) closed mesh = entry..exit hit span. The
/// span is robust against the extra tangential hits rib grooves produce
/// (an even/odd pairing can drop the main chord there).
#[cfg(test)]
fn chord_from_hits(ts: &mut Vec<f32>) -> f32 {
    dedupe_hits(ts);
    match ts.as_slice() {
        [f, .., l] => l - f,
        _ => 0.0,
    }
}

fn hash01(x: u32) -> u32 {
    let mut x = x.wrapping_mul(0x9E3779B9);
    x ^= x >> 16;
    x.wrapping_mul(0x85EBCA6B)
}

// ---------------------------------------------------------------------------
// app state
// ---------------------------------------------------------------------------

const RES_OPTIONS: [usize; 5] = [128, 160, 192, 224, 256];

// scratch state reused between detector frames
struct Scratch {
    stamp: u32,
    tested: Vec<u32>,
    hits: Vec<(u8, f32)>,
    ts: [Vec<f32>; 3],
}

// beam / imaging parameters for render_detector_image
struct Beam<'a> {
    kv: f32,
    ma: f32,
    mu_flesh: f32,
    carvings: &'a [Carving],
}

fn render_detector_image(
    grid: &Grid,
    seeds: &[(V3, V3)],
    yaw: f32,
    frame: u32,
    res: usize,
    beam: &Beam<'_>,
    sc: &mut Scratch,
) -> ColorImage {
    let c = yaw.cos();
    let s = yaw.sin();
    let mu_scale = (100.0 / beam.kv).powi(2).clamp(0.25, 6.0);
    let mu_flesh = beam.mu_flesh * mu_scale;
    let mu_seed = MU_SEED * mu_scale;
    let mu_stem = MU_STEM * mu_scale;
    let noise_lvl = 0.05 * (5.0 / beam.ma.max(0.3)).sqrt();
    // exposure gain: the detector flood is proportional to tube current
    let gain = (beam.ma / 5.0).sqrt().clamp(0.15, 2.0);
    let src0 = V3::new(SRC_X, 0.0, 0.0);
    let center_len = DET_X - SRC_X;

    let mut pixels = Vec::with_capacity(res * res);
    for iy in 0..res {
        let v = ((iy as f32 + 0.5) / res as f32) * 2.0 - 1.0;
        for ix in 0..res {
            let u = ((ix as f32 + 0.5) / res as f32) * 2.0 - 1.0;
            // detector x-axis -> world z (rotates with the model),
            // detector y-axis -> world y, flipped so the stem reads upward
            let target = V3::new(DET_X, -v * DET_HALF, u * DET_HALF);
            let d_raw = target - src0;
            let l_raw = d_raw.len();
            let d0 = d_raw.scale(1.0 / l_raw);
            // apply inverse model rotation to source & ray
            let src = src0.rot_y_inv(c, s);
            let d = d0.rot_y_inv(c, s);

            // ---- mesh attenuation (moller-trumbore through voxel grid)
            sc.stamp += 1; // per-ray stamp: prevents cross-ray triangle skipping
            sc.hits.clear();
            grid.trace(src, d, sc.stamp, &mut sc.tested, &mut sc.hits);
            sc.hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            sc.ts[0].clear();
            sc.ts[1].clear();
            sc.ts[2].clear();
            for &(g, t) in &sc.hits {
                sc.ts[g as usize].push(t);
            }
            // merge shared-edge duplicate hits, then take entry..exit spans
            for g in sc.ts.iter_mut() {
                dedupe_hits(g);
            }
            // endpoints (entry, exit) per group; chord = entry..exit pair span
            let (b0, b1) = match sc.ts[0].as_slice() {
                [f, .., l] => (*f, *l),
                _ => (0.0, 0.0),
            };
            let (c0, c1) = match sc.ts[1].as_slice() {
                [f, .., l] => (*f, *l),
                _ => (0.0, 0.0),
            };
            let t_cav = if c1 > c0 { c1 - c0 } else { 0.0 };
            let t_body = if b1 > b0 { b1 - b0 } else { 0.0 };
            let t_stem = match sc.ts[2].as_slice() {
                [f, .., l] => l - f,
                _ => 0.0,
            };

            let mut atten = mu_flesh * (t_body - t_cav).max(0.0) + mu_stem * t_stem;

            // ---- carved openings: remove flesh along the ray where the
            // carving prism overlaps the annulus between body and cavity
            if b1 > b0 {
                let flesh_a = if c0 > b0 && c1 > b0 { (b0, c0) } else { (b0, b1) };
                let flesh_b = if c0 > b0 && c1 < b1 && c1 > c0 {
                    (c1, b1)
                } else {
                    (b1, b1)
                };
                for carv in beam.carvings {
                    if let Some((t0, t1)) = carve_interval(src, d, carv) {
                        let ov_a = (flesh_a.1.min(t1) - flesh_a.0.max(t0)).max(0.0);
                        let ov_b = (flesh_b.1.min(t1) - flesh_b.0.max(t0)).max(0.0);
                        atten -= mu_flesh * (ov_a + ov_b);
                    }
                }
            }

            // ---- seeds (analytic quadrics)
            for &(center, radii) in seeds {
                let ch = ellipsoid_chord(src, d, center, radii);
                atten += mu_seed * ch;
            }
            atten = atten.max(0.0);

            // ---- image formation
            let val = (-atten).exp();
            let mut flood = (center_len / l_raw).powi(2); // inverse-square
            let r2 = u * u + v * v;
            flood *= 1.0 - 0.20 * (r2 / 2.0); // detector vignette
            if ix % 24 == 0 || iy % 24 == 0 {
                flood *= 0.96; // anti-scatter grid shadow
            }

            let mut inten = 255.0 * val * flood * gain;

            // film grain (photon noise; higher mA -> less noise)
            if inten > 1.0 {
                let sd = noise_lvl * inten.sqrt();
                let h1 = hash01((ix as u32) | ((iy as u32) << 12)) as f32 / u32::MAX as f32;
                let h2 = hash01((ix as u32).wrapping_add(frame)) as f32 / u32::MAX as f32;
                let h3 = hash01((iy as u32).wrapping_add(frame)) as f32 / u32::MAX as f32;
                let g = (h1 + h2 + h3) / 1.5 - 1.0; // ~[-1, 1]
                inten += g * sd * 2.0;
            }
            inten = inten.clamp(0.0, 255.0);
            let g = inten as u8;
            // warm film tint: slight red lift, blue drop
            let r = (inten as u32 + inten as u32 / 16).min(255) as u8;
            let b = (inten * 0.87) as u8;
            pixels.push(Color32::from_rgb(r, g, b));
        }
    }
    ColorImage {
        size: [res, res],
        pixels,
    }
}

struct App {
    grid: Grid,
    seeds: Vec<(V3, V3)>,
    carvings: Vec<Carving>,

    tex: Option<TextureHandle>,
    res_choice: usize,

    yaw: f32,
    time: f32,
    speed: f32, // deg / s
    auto: bool,

    kv: f32,
    ma: f32,
    mu_flesh: f32,
    beam_on: bool,

    frame: u32,
    fps: f32,
    last_t: Option<Instant>,

    // detector update cadence: the CPU render is throttled to keep the UI
    // responsive (also makes debug builds usable)
    last_det: Option<Instant>,
    det_ms: f32,

    sc: Scratch,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
        let mut tris = Vec::new();
        lathe(96, 48, 1.0, 1.0, 0, &mut tris); // body surface
        lathe(56, 28, 0.78, 0.85, 1, &mut tris); // hollow cavity
        stem(&mut tris);
        let grid = Grid::build(tris, 26, 1.3);
        let tested = vec![0u32; grid.tris.len()];
        let s = seeds();
        let cars = carvings();
        Self {
            grid,
            seeds: s,
            carvings: cars,
            tex: None,
            res_choice: 2,
            yaw: 0.0,
            time: 0.0,
            speed: 9.0,
            auto: true,
            kv: 100.0,
            ma: 5.0,
            mu_flesh: MU_FLESH_BASE,
            beam_on: true,
            frame: 0,
            fps: 60.0,
            last_t: None,
            last_det: None,
            det_ms: 0.0,
            sc: Scratch {
                stamp: 0,
                tested,
                hits: Vec::new(),
                ts: [Vec::new(), Vec::new(), Vec::new()],
            },
        }
    }

    // -- x-ray projector ----------------------------------------------------

    fn render_xray(&mut self) -> ColorImage {
        self.frame += 1;
        render_detector_image(
            &self.grid,
            &self.seeds,
            self.yaw,
            self.frame,
            RES_OPTIONS[self.res_choice],
            &Beam {
                kv: self.kv,
                ma: self.ma,
                mu_flesh: self.mu_flesh,
                carvings: &self.carvings,
            },
            &mut self.sc,
        )
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = self
            .last_t
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.016)
            .clamp(0.001, 0.25);
        self.last_t = Some(now);
        self.time += dt;
        if self.auto {
            self.yaw += self.speed * dt * PI / 180.0;
        }
        self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1;

        // adaptive cadence: render the CPU x-ray image at most every
        // max(33 ms, 1.5x its own cost); keeps the UI responsive even in
        // debug builds or at high detector resolutions
        let since_det = self
            .last_det
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(1.0);
        let wait = ((self.det_ms / 1000.0) * 1.5).clamp(0.033, 0.25);
        if self.beam_on && since_det >= wait {
            let t0 = Instant::now();
            let image = self.render_xray();
            self.det_ms = self.det_ms * 0.9 + t0.elapsed().as_secs_f32() * 1000.0 * 0.1;
            let res = RES_OPTIONS[self.res_choice];
            let need_new = self
                .tex
                .as_ref()
                .map(|t| t.size() != [res, res])
                .unwrap_or(true);
            if need_new {
                self.tex = Some(ctx.load_texture("xray", image, TextureOptions::LINEAR));
            } else if let Some(t) = self.tex.as_mut() {
                t.set(image, TextureOptions::LINEAR);
            }
            self.last_det = Some(now);
        }

        self.top_bar(ctx);
        self.bottom_bar(ctx);
        self.center(ctx);
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

// ---------------------------------------------------------------------------
// ui
// ---------------------------------------------------------------------------

impl App {
    fn top_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top")
            .frame(egui::Frame::NONE.fill(PANEL_BG).inner_margin(6.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("HW-XR 2026")
                            .strong()
                            .size(18.0)
                            .color(ACCENT),
                    );
                    ui.label(
                        egui::RichText::new("C-ARM PUMPKIN RADIOGRAPHY — HALLOWEEN SHIPMENT QA")
                            .size(12.0)
                            .color(TEXT_DIM),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // rightmost first: X-RAY status and FPS swapped per QA note
                        ui.label(
                            egui::RichText::new(format!("{:>3.0} FPS", self.fps))
                                .monospace()
                                .size(12.0)
                                .color(TEXT_DIM),
                        );
                        let blink = self.beam_on && (self.time * 2.0) as u32 % 2 == 0;
                        ui.label(
                            egui::RichText::new(if self.beam_on {
                                if blink { "● X-RAY ON" } else { "  X-RAY ON" }
                            } else {
                                "· X-RAY OFF"
                            })
                            .size(12.0)
                            .color(if blink {
                                Color32::from_rgb(90, 220, 90)
                            } else if self.beam_on {
                                ACCENT_DIM
                            } else {
                                TEXT_DIM
                            }),
                        );
                    });
                });
            });
    }

    fn bottom_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("bottom")
            .frame(egui::Frame::NONE.fill(PANEL_BG).inner_margin(6.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "SIDE 90 cm · MAG ×2.0 · CMOS {:04}×{:04} · {} µm PITCH · kV {} · mA {:.1}",
                            RES_OPTIONS[self.res_choice],
                            RES_OPTIONS[self.res_choice],
                            det_pitch_um(RES_OPTIONS[self.res_choice]),
                            self.kv as u32,
                            self.ma
                        ))
                        .monospace()
                        .size(11.0)
                        .color(TEXT_DIM),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new("⚠ INTERLOCKED — DO NOT ENTER BAY")
                                .size(11.0)
                                .color(Color32::from_rgb(255, 90, 60)),
                        );
                        ui.label(
                            egui::RichText::new(format!("FRAME {:06}", self.frame))
                                .monospace()
                                .size(11.0)
                                .color(TEXT_DIM),
                        );
                    });
                });
            });
    }

    fn center(&mut self, ctx: &egui::Context) {
        // controls sidebar
        egui::SidePanel::left("controls")
            .frame(egui::Frame::NONE.fill(PANEL_BG).inner_margin(8.0))
            .show(ctx, |ui| {
                ui.label(egui::RichText::new("CONSOLE").strong().color(ACCENT).size(13.0));
                ui.separator();
                    egui::ComboBox::from_label("Detector")
                    .selected_text(format!("{} µm", det_pitch_um(RES_OPTIONS[self.res_choice])))
                    .show_ui(ui, |ui| {
                        for (i, r) in RES_OPTIONS.iter().enumerate() {
                            if ui
                                .selectable_label(
                                    self.res_choice == i,
                                    format!("{} µm", det_pitch_um(*r)),
                                )
                                .clicked()
                            {
                                self.res_choice = i;
                            }
                        }
                    });
                ui.add(
                    egui::Slider::new(&mut self.speed, 0.0..=30.0)
                        .text("Rotation")
                        .suffix(" °/s"),
                );
                ui.checkbox(&mut self.auto, "Auto-rotate");
                ui.separator();
                // X-ray beam toggle
                let label = if self.beam_on {
                    egui::RichText::new("◉  X-RAY ON").strong().color(ACCENT)
                } else {
                    egui::RichText::new("○  X-RAY OFF").strong().color(TEXT_DIM)
                };
                if ui
                    .add(
                        egui::Button::new(label)
                            .fill(if self.beam_on {
                                Color32::from_rgb(70, 36, 8)
                            } else {
                                Color32::from_rgb(34, 34, 37)
                            })
                            .min_size(vec2(ui.available_width(), 26.0)),
                    )
                    .clicked()
                {
                    self.beam_on = !self.beam_on;
                }
                ui.separator();
                ui.add(egui::Slider::new(&mut self.kv, 60.0..=160.0).text("Tube kV"));
                ui.add(
                    egui::Slider::new(&mut self.ma, 0.4..=10.0)
                        .text("Tube mA")
                        .suffix(" mA"),
                );
                ui.add(
                    egui::Slider::new(&mut self.mu_flesh, 0.5..=1.6)
                        .text("Flesh μ (a.u.)")
                        .show_value(true),
                );
                ui.separator();
                ui.label(
                    egui::RichText::new(
                        "Drag the optical preview to rotate.\nAir reads bright; dense material darkens.",
                    )
                    .size(11.0)
                    .color(TEXT_DIM),
                );
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(10, 10, 11)).inner_margin(10.0))
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let side = (avail.y - 40.0).min(avail.x - 330.0).clamp(240.0, 640.0);
                let pre_size = (avail.y - 90.0).clamp(180.0, 300.0);
                ui.horizontal(|ui| {
                    // ---- optical preview
                    ui.vertical(|ui| {
                        let (prect, presp) =
                            ui.allocate_exact_size(vec2(pre_size, pre_size), Sense::click_and_drag());
                        if presp.dragged() {
                            self.yaw += presp.drag_delta().x * 0.009;
                        }
                        self.draw_preview(ui.painter(), prect);
                        let cy = (self.yaw * 180.0 / PI).rem_euclid(360.0);
                        ui.monospace(format!("OPTICAL · YAW {:4.0}°", cy));
                    });
                    ui.add_space(16.0);
                    // ---- detector image
                    ui.vertical(|ui| {
                        let resp = ui.add(egui::Image::new((tex_id(&self.tex), Vec2::splat(side))));
                        let rect = resp.rect;
                        // beam-off standby overlay over the detector image
                        if !self.beam_on {
                            ui.painter().rect_filled(
                                rect,
                                0.0,
                                Color32::from_black_alpha(170),
                            );
                            let c = rect.center();
                            ui.painter().text(
                                pos2(c.x, c.y - 10.0),
                                Align2::CENTER_CENTER,
                                "X-RAYS OFF",
                                FontId::proportional(20.0),
                                TEXT_DIM,
                            );
                            ui.painter().text(
                                pos2(c.x, c.y + 14.0),
                                Align2::CENTER_CENTER,
                                "TUBE ON STANDBY — NO SIGNAL",
                                FontId::monospace(11.0),
                                Color32::from_rgb(90, 90, 96),
                            );
                        }
                        self.draw_hud(ui, rect);
                        ui.add_space(4.0); // drop the caption below the image edge
                        ui.monospace("DETECTOR — 2D RADIOGRAPH");
                    });
                });
            });
    }

    fn draw_preview(&self, painter: &egui::Painter, rect: egui::Rect) {
        let s = 0.40 * rect.width();
        let mid = rect.center();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let pitch = -0.30f32;
        let (pc, ps) = (pitch.cos(), pitch.sin());
        let light = V3::new(0.35, 0.62, 0.70).normalized();

        // ground shadow (soft ellipse polygon)
        let mut shadow = egui::Mesh::default();
        let shadow_c = pos2(mid.x, rect.max.y - 0.04 * rect.height());
        let (rx, ry) = (0.68 * s, 0.14 * s);
        let mut pts = Vec::with_capacity(14);
        for i in 0..14 {
            let a = TAU * i as f32 / 14.0;
            pts.push(pos2(
                shadow_c.x + rx * a.cos(),
                shadow_c.y + ry * a.sin(),
            ));
        }
        let first = shadow.vertices.len() as u32;
        for p in &pts {
            shadow.colored_vertex(*p, Color32::from_black_alpha(120));
        }
        for i in 1..(pts.len() - 1) as u32 {
            shadow.add_triangle(first, first + i, first + i + 1);
        }
        painter.add(egui::Shape::mesh(shadow));

        // shaded rotating mesh, painter's algorithm (far -> near)
        let project = |v: V3| -> (Vec2, f32) {
            let r = v.rot_y(cy, sy).rot_x(pc, ps);
            (
                vec2(mid.x + r.x * s, mid.y - r.y * s),
                r.z,
            )
        };
        let mut order: Vec<(f32, usize)> = Vec::new();
        for (i, t) in self.grid.tris.iter().enumerate() {
            if t.group == 1 {
                continue; // cavity invisible from outside
            }
            let (_, z) = project(t.a);
            order.push((z, i));
        }
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut mesh = egui::Mesh::default();
        for &(_, i) in &order {
            let t = &self.grid.tris[i];
            let (pa, za) = project(t.a);
            let (pb, zb) = project(t.b);
            let (pcc, zc) = project(t.c);
            let n = ((t.b - t.a).cross(t.c - t.a)).normalized();
            let d = n.dot(light);
            let base = match t.group {
                2 => [110.0, 78.0, 34.0],
                _ => [238.0, 126.0, 28.0],
            };
            let b = (0.24 + 0.76 * d.max(0.0)).clamp(0.0, 1.0);
            let col = Color32::from_rgb(
                (base[0] * b) as u8,
                (base[1] * b) as u8,
                (base[2] * b) as u8,
            );
            let _ = (za, zb, zc);
            let base_i = mesh.vertices.len() as u32;
            mesh.colored_vertex(pos2(pa.x, pa.y), col);
            mesh.colored_vertex(pos2(pb.x, pb.y), col);
            mesh.colored_vertex(pos2(pcc.x, pcc.y), col);
            mesh.add_triangle(base_i, base_i + 1, base_i + 2);
        }
        painter.add(egui::Shape::mesh(mesh));

        // ---- inner light bleeding through the carved openings
        let view_front = V3::new(0.0, 0.0, 1.0); // camera looks from +z in view space
        for c in &self.carvings {
            let n_w = c.n.rot_y(cy, sy).rot_x(pc, ps);
            let facing = n_w.dot(view_front);
            if facing <= 0.02 {
                continue; // opening points away from the camera
            }
            let pulse = 0.5 + 0.5 * (self.time * 2.2).sin();
            let glow = 0.35 + 0.65 * facing;
            let bright = (glow * (0.75 + 0.25 * pulse)).clamp(0.0, 1.0);
            let a = (70.0 + 185.0 * bright) as u8;
            // flame flicker between deep orange and yellow-white
            let r = 255u8;
            let g = (110.0 + 110.0 * bright) as u8;
            let b = (10.0 + 90.0 * bright) as u8;
            let col = Color32::from_rgba_premultiplied(
                (r as f32 * (a as f32 / 255.0)) as u8,
                (g as f32 * (a as f32 / 255.0)) as u8,
                (b as f32 * (a as f32 / 255.0)) as u8,
                a,
            );
            // hug the curved skin: every corner sits on the surface (a tangential
            // triangle would lift off and float in the air)
            let mut gm = egui::Mesh::default();
            let base = gm.vertices.len() as u32;
            for &(du, dv) in &c.tri {
                let dir = (c.anchor + c.u.scale(du) + c.v.scale(dv)).normalized();
                let p = surface_point(dir) + c.n.scale(0.012); // just above the skin
                let (sp, _) = project(p);
                gm.colored_vertex(pos2(sp.x, sp.y), col);
            }
            gm.add_triangle(base, base + 1, base + 2);
            painter.add(egui::Shape::mesh(gm));
        }
    }

    fn draw_hud(&self, ui: &egui::Ui, rect: egui::Rect) {
        let p = ui.painter();
        // frame + corner ticks
        for (a, b) in [
            (pos2(rect.min.x - 6.0, rect.min.y - 6.0), pos2(rect.max.x + 6.0, rect.min.y - 6.0)),
            (pos2(rect.min.x - 6.0, rect.max.y + 6.0), pos2(rect.max.x + 6.0, rect.max.y + 6.0)),
            (pos2(rect.min.x - 6.0, rect.min.y - 6.0), pos2(rect.min.x - 6.0, rect.max.y + 6.0)),
            (pos2(rect.max.x + 6.0, rect.min.y - 6.0), pos2(rect.max.x + 6.0, rect.max.y + 6.0)),
        ] {
            p.line_segment([a, b], (1.0, ACCENT_DIM));
        }
        let tick = 14.0;
        for (c, (dx, dy)) in [
            (rect.min, (1.0, 1.0)),
            (rect.max, (-1.0, -1.0)),
            (pos2(rect.min.x, rect.max.y), (1.0, -1.0)),
            (pos2(rect.max.x, rect.min.y), (-1.0, 1.0)),
        ] {
            p.line_segment(
                [c, pos2(c.x + dx * tick, c.y)],
                (2.0, ACCENT),
            );
            p.line_segment(
                [c, pos2(c.x, c.y + dy * tick)],
                (2.0, ACCENT),
            );
        }
        // labels
        let mono = |s: String, al: Align2, pos: egui::Pos2| {
            p.text(pos, al, s, FontId::monospace(10.0), TEXT_DIM);
        };
        let pad = 10.0;
        // blinking X-ray warning sign (vector trefoil), top center of the detector
        if self.beam_on {
            let blink = (self.time * 3.0) as u32 % 2 == 0;
            let col = if blink { ACCENT } else { Color32::from_rgb(200, 60, 30) };
            draw_trefoil(p, pos2(rect.center().x, rect.min.y + pad + 9.0), 9.0, col);
            p.text(
                pos2(rect.center().x, rect.min.y + pad + 20.0),
                Align2::CENTER_TOP,
                "X-RAY",
                FontId::proportional(15.0),
                col,
            );
        } else {
            p.text(
                pos2(rect.center().x, rect.min.y + pad),
                Align2::CENTER_TOP,
                "STANDBY",
                FontId::proportional(11.0),
                TEXT_DIM,
            );
        }
        mono("SID 90 cm · MAG ×2.0".into(), Align2::LEFT_TOP, pos2(rect.min.x + pad, rect.min.y + pad));
        mono(format!("FRM {:06}", self.frame), Align2::RIGHT_TOP, pos2(rect.max.x - pad, rect.min.y + pad));
        mono(
            format!("{} kV · {:.1} mA", self.kv as u32, self.ma),
            Align2::LEFT_BOTTOM,
            pos2(rect.min.x + pad, rect.max.y - pad),
        );
        mono(
            format!("PITCH {} µm", det_pitch_um(RES_OPTIONS[self.res_choice])),
            Align2::RIGHT_BOTTOM,
            pos2(rect.max.x - pad, rect.max.y - pad),
        );
    }
}

fn tex_id(tex: &Option<TextureHandle>) -> egui::TextureId {
    tex.as_ref()
        .map(|t| t.id())
        .unwrap_or(egui::TextureId::default())
}

/// Ionizing-radiation trefoil drawn as vector geometry (the glyph may be
/// missing from egui's default fonts). `c` = center of the inner hub,
/// `r` = outer blade radius.
fn draw_trefoil(p: &egui::Painter, c: egui::Pos2, r: f32, col: Color32) {
    let mut mesh = egui::Mesh::default();
    let add_tri = |m: &mut egui::Mesh, a: egui::Pos2, b: egui::Pos2, d: egui::Pos2| {
        let base = m.vertices.len() as u32;
        for v in [a, b, d] {
            m.colored_vertex(v, col);
        }
        m.add_triangle(base, base + 1, base + 2);
    };
    // hub disc (fan of triangles, ~0.32 r)
    let hr = 0.32 * r;
    for k in 0..20 {
        let a0 = (k as f32 / 20.0) * std::f32::consts::TAU;
        let a1 = ((k + 1) as f32 / 20.0) * std::f32::consts::TAU;
        add_tri(
            &mut mesh,
            c,
            c + vec2(hr * a0.cos(), hr * a0.sin()),
            c + vec2(hr * a1.cos(), hr * a1.sin()),
        );
    }
    // 3 blades: 60° sectors of the annulus 0.35r..r, rotated -90°, 30°, 150°
    for blade in 0..3 {
        let base_ang = (-90.0 + 120.0 * blade as f32).to_radians();
        let steps = 12; // arc tessellation per 60° sector
        let inner = 0.35 * r;
        for k in 0..steps {
            let t0 = base_ang + (k as f32 / steps as f32) * (60f32.to_radians());
            let t1 = base_ang + ((k + 1) as f32 / steps as f32) * (60f32.to_radians());
            let o0 = c + vec2(inner * t0.cos(), inner * t0.sin());
            let o1 = c + vec2(inner * t1.cos(), inner * t1.sin());
            let o2 = c + vec2(r * t1.cos(), r * t1.sin());
            let o3 = c + vec2(r * t0.cos(), r * t0.sin());
            add_tri(&mut mesh, o0, o1, o2);
            add_tri(&mut mesh, o0, o2, o3);
        }
    }
    p.add(egui::Shape::mesh(mesh));
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "writes target/xray_frame.png for manual inspection"]
    fn dump_xray_frame_png() {
        let mut tris = Vec::new();
        lathe(64, 32, 1.0, 1.0, 0, &mut tris);
        lathe(40, 20, 0.78, 0.85, 1, &mut tris);
        stem(&mut tris);
        let grid = Grid::build(tris, 26, 1.3);
        let seeds = seeds();
        let cars = carvings();
        let mut sc = Scratch {
            stamp: 0,
            tested: vec![0; grid.tris.len()],
            hits: Vec::new(),
            ts: [Vec::new(), Vec::new(), Vec::new()],
        };
        let img = render_detector_image(
            &grid,
            &seeds,
            0.0,
            1,
            224,
            &Beam {
                kv: 100.0,
                ma: 5.0,
                mu_flesh: MU_FLESH_BASE,
                carvings: &cars,
            },
            &mut sc,
        );
        let res = img.size[0];
        // block-mean map (16x16 blocks) for quick visual inspection
        let blocks = 16;
        let bs = res / blocks;
        let mut map = String::new();
        for by in 0..blocks {
            for bx in 0..blocks {
                let mut sum = 0.0f32;
                for yy in 0..bs {
                    for xx in 0..bs {
                        sum += img.pixels[(by * bs + yy) * res + bx * bs + xx].r() as f32;
                    }
                }
                let m = sum / (bs * bs) as f32;
                let digit = ((m / 255.0) * 9.0).round() as u8;
                map.push_str(&format!("{digit}"));
            }
            map.push('\n');
        }
        eprintln!("dump: block means (0=dark, 9=bright):\n{map}");

        let mut rgba = image::RgbaImage::new(res as u32, res as u32);
        let px = |x: usize, y: usize| img.pixels[y * res + x];
        for y in 0..res {
            for x in 0..res {
                let c = px(x, y);
                rgba.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgba([c.r(), c.g(), c.b(), 255]),
                );
            }
        }
        rgba.save("target/xray_frame.png").expect("save png");
        eprintln!("dump: wrote target/xray_frame.png");
    }

    #[test]
    fn xray_equator_chords_are_sane() {
        let mut tris = Vec::new();
        lathe(64, 32, 1.0, 1.0, 0, &mut tris);
        lathe(40, 20, 0.78, 0.85, 1, &mut tris);
        stem(&mut tris);
        let grid = Grid::build(tris, 26, 1.3);
        let mut tested = vec![0u32; grid.tris.len()];
        let mut hits = Vec::new();

        // ray parallel to +x, slightly off the exact equator seam to avoid
        // hitting the degenerate y=z=0 cell boundary line
        let src = V3::new(SRC_X, 0.03, 0.0);
        let d = V3::new(1.0, 0.0, 0.0);
        grid.trace(src, d, 1, &mut tested, &mut hits);
        assert!(hits.len() >= 8, "too few mesh hits: {}", hits.len());

        let mut per_g = [Vec::new(), Vec::new(), Vec::new()];
        for &(g, t) in &hits {
            per_g[g as usize].push(t);
        }
        let body = chord_from_hits(&mut per_g[0]);
        let cav = chord_from_hits(&mut per_g[1]);
        // entry+exit of the ribbed skin shell (theta=0 -> ~1.01, theta=pi -> ~0.87)
        assert!(body > 1.6 && body < 2.2, "body chord {body}");
        // cavity chord is shorter (hollow core)
        assert!(cav > 1.0 && cav < 1.7, "cavity chord {cav}");
        assert!(cav < body);
        // the stem sits on top; the equator ray must not cross it
        assert!(per_g[2].is_empty());
    }

    #[test]
    fn air_pixel_has_no_attenuation() {
        let mut tris = Vec::new();
        lathe(64, 32, 1.0, 1.0, 0, &mut tris);
        lathe(40, 20, 0.78, 0.85, 1, &mut tris);
        stem(&mut tris);
        let grid = Grid::build(tris, 26, 1.3);
        let mut tested = vec![0u32; grid.tris.len()];
        let mut hits = Vec::new();
        let src = V3::new(SRC_X, 0.0, 0.0);
        // ray that misses the pumpkin entirely (well above it)
        let target = V3::new(DET_X, 0.0, 2.3);
        let d = (target - src).normalized();
        grid.trace(src, d, 1, &mut tested, &mut hits);
        assert!(hits.is_empty(), "expected no hits for an air ray");
    }
}

// ---------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 780.0])
            .with_min_inner_size([980.0, 640.0])
            .with_title("HW-XR 2026 · C-Arm Pumpkin Radiography"),
        ..Default::default()
    };
    eframe::run_native(
        "HW-XR 2026",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)) as Box<dyn eframe::App>)),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;

    console_error_panic_hook::set_once();
    let web_options = eframe::WebOptions::default();
    wasm_bindgen_futures::spawn_local(async move {
        let document = eframe::web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .create_element("canvas")
            .unwrap()
            .dyn_into::<eframe::web_sys::HtmlCanvasElement>()
            .unwrap();
        // fill the whole viewport: without CSS sizing the canvas defaults to a
        // 300x150 CSS-px drawing area, so the GUI shows small and clipped
        let style = canvas.style();
        style.set_property("display", "block").unwrap();
        style.set_property("width", "100%").unwrap();
        style.set_property("height", "100%").unwrap();
        document.body().unwrap().append_child(&canvas).unwrap();
        // app booted: drop the trunk loading placeholder
        let _ = document
            .get_element_by_id("loading")
            .map(|el| el.remove());
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(App::new(cc)) as Box<dyn eframe::App>)),
            )
            .await
            .expect("eframe failed to start");
    });
}
