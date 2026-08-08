//! A depth-buffered character canvas that emits ANSI truecolor.
//!
//! The observatory composites several 3D views into one frame, so the canvas is
//! z-buffered: a cell is only overwritten by a fragment that is nearer to the
//! camera, which is what lets the tornado occlude itself instead of painting the
//! last-drawn particle on top.

use std::fmt::Write as _;

use crate::theme::{self, Rgb};

/// One character cell.
#[derive(Debug, Clone, Copy)]
struct Cell {
    ch: char,
    fg: Rgb,
    /// Camera-space depth; smaller is nearer. `f64::INFINITY` means empty.
    depth: f64,
}

impl Cell {
    const EMPTY: Cell = Cell {
        ch: ' ',
        fg: theme::TEXT,
        depth: f64::INFINITY,
    };
}

/// Where and how large a 3D view is drawn: centre in cells, plus the extent it
/// is allowed to occupy. Bundling these keeps the draw calls readable and lets
/// the tornado size itself to its panel.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub cx: f64,
    pub cy: f64,
    pub w: usize,
    pub h: usize,
}

impl Viewport {
    pub fn new(cx: f64, cy: f64, w: usize, h: usize) -> Self {
        Self { cx, cy, w, h }
    }
}

/// A fixed-size drawing surface.
#[derive(Debug, Clone)]
pub struct Canvas {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
    /// Active clip region as `(x0, y0, x1, y1)`, inclusive. Writes outside it are
    /// dropped, which is what keeps a panel's labels inside its own border.
    clip: (i64, i64, i64, i64),
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        Self {
            width,
            height,
            cells: vec![Cell::EMPTY; width * height],
            clip: (0, 0, width as i64 - 1, height as i64 - 1),
        }
    }

    /// Restrict drawing to a rectangle, returning the previous clip so the
    /// caller can restore it. Panels use this so their contents can never bleed
    /// across a border into a neighbour.
    pub fn set_clip(&mut self, x: i64, y: i64, w: usize, h: usize) -> (i64, i64, i64, i64) {
        let previous = self.clip;
        let x1 = x + w as i64 - 1;
        let y1 = y + h as i64 - 1;
        self.clip = (
            x.max(0),
            y.max(0),
            x1.min(self.width as i64 - 1),
            y1.min(self.height as i64 - 1),
        );
        previous
    }

    /// Restore a clip returned by [`Canvas::set_clip`].
    pub fn restore_clip(&mut self, clip: (i64, i64, i64, i64)) {
        self.clip = clip;
    }

    /// Reset the clip to the whole canvas.
    pub fn reset_clip(&mut self) {
        self.clip = (0, 0, self.width as i64 - 1, self.height as i64 - 1);
    }

    fn in_clip(&self, x: i64, y: i64) -> bool {
        let (x0, y0, x1, y1) = self.clip;
        x >= x0 && x <= x1 && y >= y0 && y <= y1
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn clear(&mut self) {
        self.cells.fill(Cell::EMPTY);
        self.reset_clip();
    }

    fn index(&self, x: usize, y: usize) -> Option<usize> {
        if x < self.width && y < self.height {
            Some(y * self.width + x)
        } else {
            None
        }
    }

    /// Draw a fragment, honouring the depth buffer. Out-of-bounds is a no-op, so
    /// callers may project freely without clipping first.
    pub fn put_depth(&mut self, x: i64, y: i64, ch: char, fg: Rgb, depth: f64) {
        if x < 0 || y < 0 || depth.is_nan() || !self.in_clip(x, y) {
            return;
        }
        let Some(index) = self.index(x as usize, y as usize) else {
            return;
        };
        if depth <= self.cells[index].depth {
            self.cells[index] = Cell { ch, fg, depth };
        }
    }

    /// Draw a fragment at the front of the buffer — for UI chrome and text,
    /// which should never be occluded by geometry.
    pub fn put(&mut self, x: i64, y: i64, ch: char, fg: Rgb) {
        self.put_depth(x, y, ch, fg, f64::NEG_INFINITY);
    }

    /// Draw a string left-to-right from `x`, clipped at the right edge.
    pub fn text(&mut self, x: i64, y: i64, s: &str, fg: Rgb) {
        for (offset, ch) in s.chars().enumerate() {
            self.put(x + offset as i64, y, ch, fg);
        }
    }

    /// Draw a string only where it fits within `max_width` columns, truncating
    /// with an ellipsis rather than spilling into the neighbouring panel.
    pub fn text_clipped(&mut self, x: i64, y: i64, s: &str, fg: Rgb, max_width: usize) {
        if max_width == 0 {
            return;
        }
        let count = s.chars().count();
        if count <= max_width {
            self.text(x, y, s, fg);
            return;
        }
        let keep = max_width.saturating_sub(1);
        let truncated: String = s.chars().take(keep).collect();
        self.text(x, y, &truncated, fg);
        self.put(x + keep as i64, y, '…', fg);
    }

    /// Bresenham line from `(x0, y0)` to `(x1, y1)` at a constant depth.
    #[allow(clippy::too_many_arguments)]
    pub fn line(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, ch: char, fg: Rgb, depth: f64) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x0, y0);

        loop {
            self.put_depth(x, y, ch, fg, depth);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }

    /// A single-line box with an optional title in the top rule.
    pub fn frame(&mut self, x: i64, y: i64, w: usize, h: usize, title: &str, fg: Rgb) {
        if w < 2 || h < 2 {
            return;
        }
        let right = x + w as i64 - 1;
        let bottom = y + h as i64 - 1;

        for column in x + 1..right {
            self.put(column, y, '─', fg);
            self.put(column, bottom, '─', fg);
        }
        for row in y + 1..bottom {
            self.put(x, row, '│', fg);
            self.put(right, row, '│', fg);
        }
        self.put(x, y, '╭', fg);
        self.put(right, y, '╮', fg);
        self.put(x, bottom, '╰', fg);
        self.put(right, bottom, '╯', fg);

        if !title.is_empty() && w > 6 {
            let label = format!(" {title} ");
            self.text_clipped(x + 2, y, &label, theme::ACCENT, w - 4);
        }
    }

    /// A horizontal meter: `filled` fraction of `w` cells, coloured by ramp.
    pub fn bar(&mut self, x: i64, y: i64, w: usize, filled: f64, ramp: impl Fn(f64) -> Rgb) {
        let filled = filled.clamp(0.0, 1.0);
        let full = filled * w as f64;
        for column in 0..w {
            let local = (full - column as f64).clamp(0.0, 1.0);
            let ch = if local <= 0.0 { '░' } else { theme::glyph(local) };
            let colour = if local <= 0.0 {
                theme::FRAME
            } else {
                ramp((column as f64 + 0.5) / w as f64)
            };
            self.put(x + column as i64, y, ch, colour);
        }
    }

    /// Render to an ANSI string. Colour changes are emitted only when the colour
    /// actually changes, which keeps a full-screen frame to a few KB.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.width * self.height * 4);
        let mut current: Option<Rgb> = None;

        for row in 0..self.height {
            for column in 0..self.width {
                let cell = self.cells[row * self.width + column];
                if current != Some(cell.fg) {
                    let Rgb(r, g, b) = cell.fg;
                    let _ = write!(out, "\x1b[38;2;{r};{g};{b}m");
                    current = Some(cell.fg);
                }
                out.push(cell.ch);
            }
            out.push_str("\x1b[0m");
            current = None;
            if row + 1 < self.height {
                out.push('\n');
            }
        }
        out
    }

    /// Render without escape codes — used by the snapshot tests and `--plain`.
    pub fn render_plain(&self) -> String {
        let mut out = String::with_capacity(self.width * self.height);
        for row in 0..self.height {
            for column in 0..self.width {
                out.push(self.cells[row * self.width + column].ch);
            }
            if row + 1 < self.height {
                out.push('\n');
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_bounds_writes_are_ignored() {
        let mut canvas = Canvas::new(4, 3);
        canvas.put(-1, 0, 'x', theme::TEXT);
        canvas.put(0, -1, 'x', theme::TEXT);
        canvas.put(99, 0, 'x', theme::TEXT);
        canvas.put(0, 99, 'x', theme::TEXT);
        assert_eq!(canvas.render_plain(), "    \n    \n    ");
    }

    #[test]
    fn nearer_fragments_win_and_farther_ones_lose() {
        let mut canvas = Canvas::new(3, 1);
        canvas.put_depth(0, 0, 'f', theme::TEXT, 10.0);
        canvas.put_depth(0, 0, 'n', theme::TEXT, 1.0);
        canvas.put_depth(0, 0, 'x', theme::TEXT, 50.0);
        assert_eq!(canvas.render_plain().chars().next(), Some('n'));
    }

    #[test]
    fn text_is_clipped_with_an_ellipsis() {
        let mut canvas = Canvas::new(10, 1);
        canvas.text_clipped(0, 0, "abcdefghijklm", theme::TEXT, 5);
        assert_eq!(canvas.render_plain(), "abcd…     ");
    }

    #[test]
    fn frame_draws_its_corners() {
        let mut canvas = Canvas::new(6, 3);
        canvas.frame(0, 0, 6, 3, "", theme::FRAME);
        let plain = canvas.render_plain();
        let rows: Vec<&str> = plain.lines().collect();
        assert!(rows[0].starts_with('╭') && rows[0].ends_with('╮'));
        assert!(rows[2].starts_with('╰') && rows[2].ends_with('╯'));
    }

    #[test]
    fn a_line_connects_both_endpoints() {
        let mut canvas = Canvas::new(5, 5);
        canvas.line(0, 0, 4, 4, '*', theme::TEXT, 0.0);
        let rows: Vec<String> = canvas.render_plain().lines().map(str::to_string).collect();
        assert_eq!(rows[0].chars().next(), Some('*'));
        assert_eq!(rows[4].chars().nth(4), Some('*'));
    }

    #[test]
    fn ansi_render_resets_every_row() {
        let canvas = Canvas::new(2, 2);
        assert_eq!(canvas.render().matches("\x1b[0m").count(), 2);
    }

    #[test]
    fn writes_outside_the_clip_region_are_dropped() {
        let mut canvas = Canvas::new(10, 3);
        canvas.set_clip(2, 1, 3, 1);
        canvas.text(0, 1, "abcdefghij", theme::TEXT);
        canvas.text(0, 0, "xxxxxxxxxx", theme::TEXT);
        let rows: Vec<&str> = {
            let plain = canvas.render_plain();
            Box::leak(plain.into_boxed_str()).lines().collect()
        };
        assert_eq!(rows[0], "          ", "the clip excluded row 0 entirely");
        assert_eq!(rows[1], "  cde     ", "only the clipped span survived");
    }

    #[test]
    fn clips_restore_and_reset() {
        let mut canvas = Canvas::new(6, 1);
        let outer = canvas.set_clip(0, 0, 2, 1);
        canvas.restore_clip(outer);
        canvas.text(0, 0, "abcdef", theme::TEXT);
        assert_eq!(canvas.render_plain(), "abcdef");

        canvas.set_clip(0, 0, 1, 1);
        canvas.clear();
        canvas.text(0, 0, "zzzzzz", theme::TEXT);
        assert_eq!(canvas.render_plain(), "zzzzzz", "clear resets the clip");
    }
}
